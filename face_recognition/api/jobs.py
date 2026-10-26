"""Background job metadata; person results retain their small public schema."""
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid


class JobQueueFull(RuntimeError):
    """Bounded inference capacity is occupied; the client may retry later."""


class JobManager:
    def __init__(self, workers=1, capacity=4, history=128):
        if not 1 <= workers <= capacity or history < 1:
            raise ValueError('Require 1 <= workers <= capacity and positive history')
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self._lock = threading.Lock()
        self._jobs = {}
        self._capacity = threading.BoundedSemaphore(capacity)
        self._history = history
        self._closed = False

    def submit(self, worker):
        with self._lock:
            if self._closed:
                raise RuntimeError('Job manager is shutting down')
            if not self._capacity.acquire(blocking=False):
                raise JobQueueFull('Video processing capacity is full; retry later')
            finished = [key for key, row in self._jobs.items() if row['status'] in {'done', 'failed'}]
            for key in finished[:max(0, len(finished) - self._history)]:
                del self._jobs[key]
            job_id = str(uuid.uuid4())
            self._jobs[job_id] = {"job_id": job_id, "status": "pending", "result": None, "error": None}
            try:
                self.executor.submit(self._run, job_id, worker)
            except Exception:
                del self._jobs[job_id]
                self._capacity.release()
                raise
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
            finished = [key for key, row in self._jobs.items() if row['status'] in {'done', 'failed'}]
            for key in finished[:max(0, len(finished) - self._history)]:
                del self._jobs[key]
        self._capacity.release()

    def get(self, job_id):
        with self._lock:
            return dict(self._jobs[job_id]) if job_id in self._jobs else None

    def close(self):
        with self._lock:
            self._closed = True
        self.executor.shutdown(wait=True)
