"""Optional HTTP API over the same services used by the two main programs."""
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import threading
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse

from face_recognition.api.jobs import JobManager, JobQueueFull
from face_recognition.api.schemas import VideoRequest
from face_recognition.config import PROJECT_ROOT, Settings
from face_recognition.core.images import decode_photo
from face_recognition.runtime import create_runtime
from face_recognition.storage.attendance import parse_timestamp
from face_recognition.video.pipeline import StreamPipeline
from face_recognition.video.results import SessionResults


def create_app(runtime_factory=create_runtime, settings=None):
    settings = settings or Settings.from_env()
    runtime = None
    runtime_lock = threading.Lock()
    jobs = JobManager(settings.job_workers, settings.job_capacity, settings.job_history)

    def get_runtime():
        nonlocal runtime
        with runtime_lock:
            if runtime is None:
                runtime = runtime_factory(settings)
            return runtime

    @asynccontextmanager
    async def lifespan(app):
        yield
        await run_in_threadpool(jobs.close)
        if runtime is not None:
            await run_in_threadpool(runtime.people.client.close)

    app = FastAPI(title="Face Recognition API", version="2.0.0", lifespan=lifespan)

    async def read_photo(upload):
        data = await upload.read(settings.max_image_bytes + 1)
        if not data or len(data) > settings.max_image_bytes:
            raise HTTPException(status_code=422, detail=f"photo must contain 1 to {settings.max_image_bytes} bytes")
        return data

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        """Load real inference assets and verify the gallery can be queried."""
        try:
            active = get_runtime()
            active.people.client.get_collection(active.people.collection)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Recognition readiness failed")
            raise HTTPException(status_code=503, detail="Recognition model or gallery unavailable") from exc
        return {"status": "ready", "model": settings.model_name,
                "device": active.backend.device,
                "storage": "embedded" if settings.qdrant_path else "server"}

    @app.post("/enroll")
    async def enroll(name: str = Form(...), image: UploadFile = File(...), person_id: str | None = Form(None)):
        data = await read_photo(image)
        def work():
            return get_runtime().enrollment().enroll(name, data, person_id)
        try:
            return await run_in_threadpool(work)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/recognize/image")
    async def recognize_image(image: UploadFile = File(...), confidence: float | None = Query(None, ge=0, le=1)):
        data = await read_photo(image)
        def work():
            frame = decode_photo(data, settings.max_image_bytes, settings.max_image_dimension)
            results = SessionResults()
            return results.observe(get_runtime().engine(confidence).recognize(frame), datetime.now().astimezone())
        try:
            return await run_in_threadpool(work)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/video-recognition", status_code=202)
    def video_recognition(request: VideoRequest):
        def work(job_id):
            output_root = settings.output_dir or PROJECT_ROOT / 'output'
            folder = output_root / "jobs" / job_id
            attendance_file = None
            if request.mode == "attendance":
                day = parse_timestamp(request.start_time).date()
                attendance_file = request.attendance_file or output_root / "attendance" / f"{day}.json"
            results = StreamPipeline(get_runtime().engine(request.confidence), settings=settings).run_video(
                request.video_path, output=folder / "results.json", start_time=request.start_time,
                sample_every=request.sample_every, action=request.action, attendance_file=attendance_file,
                annotated_video=folder / "annotated.mp4" if request.create_annotated_video else None,
            )
            return {"records": results, "output_file": str(folder / "results.json"),
                    "annotated_video": str(folder / "annotated.mp4") if request.create_annotated_video else None}
        try:
            return jobs.submit(work)
        except JobQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={'Retry-After': '2'}) from exc

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get('/jobs/{job_id}/artifacts/{kind}')
    def get_artifact(job_id: str, kind: Literal['video', 'results']):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='job not found')
        if job['status'] != 'done':
            raise HTTPException(status_code=409, detail='job has no completed artifacts')
        key = 'annotated_video' if kind == 'video' else 'output_file'
        path = job['result'].get(key)
        if path is None or not Path(path).is_file():
            raise HTTPException(status_code=404, detail='artifact not created')
        return FileResponse(path, media_type='video/mp4' if kind == 'video' else 'application/json',
                            filename=f'{job_id}.mp4' if kind == 'video' else f'{job_id}.json')

    return app


app = create_app()
