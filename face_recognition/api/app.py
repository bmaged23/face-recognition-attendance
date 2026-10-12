"""Optional HTTP API over the same services used by the two main programs."""
from contextlib import asynccontextmanager
from datetime import datetime
import threading

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from face_recognition.api.jobs import JobManager
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
    jobs = JobManager()

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

    app = FastAPI(title="Face Recognition API", version="2.0.0", lifespan=lifespan)

    async def read_photo(upload):
        data = await upload.read(settings.max_image_bytes + 1)
        if not data or len(data) > settings.max_image_bytes:
            raise HTTPException(status_code=422, detail=f"photo must contain 1 to {settings.max_image_bytes} bytes")
        return data

    @app.get("/health")
    def health():
        return {"status": "ok"}

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
            frame = decode_photo(data, settings.max_image_bytes)
            results = SessionResults()
            return results.observe(get_runtime().engine(confidence).recognize(frame), datetime.now().astimezone())
        try:
            return await run_in_threadpool(work)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/video-recognition", status_code=202)
    def video_recognition(request: VideoRequest):
        def work(job_id):
            folder = PROJECT_ROOT / "output" / "jobs" / job_id
            attendance_file = None
            if request.mode == "attendance":
                day = parse_timestamp(request.start_time).date()
                attendance_file = request.attendance_file or PROJECT_ROOT / "output" / "attendance" / f"{day}.json"
            results = StreamPipeline(get_runtime().engine(request.confidence), settings=settings).run_video(
                request.video_path, output=folder / "results.json", start_time=request.start_time,
                sample_every=request.sample_every, action=request.action, attendance_file=attendance_file,
                annotated_video=folder / "annotated.mp4" if request.create_annotated_video else None,
            )
            return {"records": results, "output_file": str(folder / "results.json"),
                    "annotated_video": str(folder / "annotated.mp4") if request.create_annotated_video else None}
        return jobs.submit(work)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    return app


app = create_app()
