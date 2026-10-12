"""Background job metadata; person results retain their small public schema."""
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid


class JobManager:
    def __init__(self, workers=2):
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self._lock = threading.Lock()
        self._jobs = {}

    def submit(self, worker):
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {"job_id": job_id, "status": "pending", "result": None, "error": None}
        self.executor.submit(self._run, job_id, worker)
        return {"job_id": job_id}

    def _run(self, job_id, worker):
        with self._lock:
            self._jobs[job_id]["status"] = "running"
        try:
            result = worker(job_id)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Video recognition job %s failed", job_id)
            update = {"status": "failed", "error": str(exc)}
        else:
            update = {"status": "done", "result": result}
        with self._lock:
            self._jobs[job_id].update(update)

    def get(self, job_id):
        with self._lock:
            return dict(self._jobs[job_id]) if job_id in self._jobs else None

    def close(self):
        self.executor.shutdown(wait=True)
