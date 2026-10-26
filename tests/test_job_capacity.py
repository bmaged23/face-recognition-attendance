"""Real worker threads: backpressure, failure recovery and bounded metadata."""
import threading
import time
import unittest
from face_recognition.api.jobs import JobManager, JobQueueFull


class JobCapacityTests(unittest.TestCase):
    def test_full_queue_rejects_work_without_losing_running_job(self):
        manager = JobManager(workers=1, capacity=1)
        started, release = threading.Event(), threading.Event()
        def work(_):
            started.set()
            if not release.wait(3):
                raise RuntimeError('Test worker was not released')
            return {'preserved': True}
        try:
            job = manager.submit(work)['job_id']
            self.assertTrue(started.wait(2))
            with self.assertRaises(JobQueueFull):
                manager.submit(lambda _: None)
            self.assertEqual(manager.get(job)['status'], 'running')
        finally:
            release.set()
            manager.close()
        self.assertEqual(manager.get(job)['result'], {'preserved': True})
        with self.assertRaisesRegex(RuntimeError, 'shutting down'):
            manager.submit(lambda _: None)

    def test_failed_job_releases_capacity_and_history_is_bounded(self):
        manager = JobManager(workers=1, capacity=1, history=1)
        def fail(_):
            raise ValueError('controlled failure')
        def wait_finished(job):
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if manager.get(job)['status'] in {'done', 'failed'}:
                    return
                time.sleep(0.005)
            self.fail('Worker did not finish')
        try:
            first = manager.submit(fail)['job_id']
            wait_finished(first)
            self.assertEqual(manager.get(first)['status'], 'failed')
            second = manager.submit(lambda _: 42)['job_id']
            wait_finished(second)
            self.assertIsNone(manager.get(first))
            self.assertEqual(manager.get(second)['result'], 42)
        finally:
            manager.close()
