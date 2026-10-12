"""Media seeking must return the requested bytes, and private files stay private."""
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from scripts.serve_demo import DemoHandler, parse_byte_range


class DemoTests(unittest.TestCase):
    def test_range_parser_handles_suffix_and_open_ended_requests(self):
        self.assertEqual(parse_byte_range('bytes=2-', 10), (2, 9))
        self.assertEqual(parse_byte_range('bytes=-3', 10), (7, 9))
        self.assertEqual(parse_byte_range('bytes=2-99', 10), (2, 9))
        for header in ('bytes=10-', 'bytes=3-1', 'bytes=-0', 'bytes=-', 'bytes=1-2,3-4'):
            with self.assertRaises(ValueError):
                parse_byte_range(header, 10)

    def test_video_seek_returns_partial_content_and_blocks_private_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            video_path = '/output/chokepoint_accuracy/improved/P2L_S5_C2_annotated.mp4'
            target = Path(folder) / video_path.lstrip('/')
            target.parent.mkdir(parents=True)
            target.write_bytes(b'0123456789')
            (Path(folder) / '.env').write_text('private')
            class QuietHandler(DemoHandler):
                def log_message(self, *args):
                    pass
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=folder))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                request = urllib.request.Request(base + video_path, headers={'Range': 'bytes=2-5'})
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers['Content-Range'], 'bytes 2-5/10')
                    self.assertEqual(response.read(), b'2345')
                request = urllib.request.Request(base + video_path, headers={'Range': 'bytes=20-'})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, 416)
                for private in ('/.env', '/docs/demo/../../../.env', '/%2eenv'):
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(base + private)
                    self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
