"""Serve only the demo page, diagram, and two approved videos on localhost."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = {
    '/output/chokepoint_accuracy/improved/P2L_S5_C2_annotated.mp4',
    '/output/chokepoint_accuracy/improved/P2E_S5_C2_annotated.mp4',
}
PAGES = {'/', '/docs/demo/index.html', '/docs/demo/playlist.js', '/docs/media/pipeline.gif', '/docs/media/pipeline.svg'}


def parse_byte_range(header, size):
    """Parse a single HTTP byte range, including open-ended and suffix ranges."""
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if not match or size <= 0:
        raise ValueError('Invalid range')
    first, last = match.groups()
    if not first:
        if not last or int(last) <= 0:
            raise ValueError('Invalid suffix range')
        return max(0, size - int(last)), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError('Range outside file')
    return start, end


class DemoHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range = None
        if unquote(urlsplit(self.path).path) not in VIDEOS:
            return super().send_head()
        path = self.translate_path(self.path)
        try:
            stream = open(path, 'rb')
        except OSError:
            self.send_error(404)
            return None
        size = Path(path).stat().st_size
        header = self.headers.get('Range')
        if header:
            try:
                start, end = parse_byte_range(header, size)
            except ValueError:
                stream.close()
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return None
            self.byte_range = start, end
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            stream.seek(start)
        else:
            start, end = 0, size - 1
            self.send_response(200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self.byte_range is None:
                return super().copyfile(source, outputfile)
            remaining = self.byte_range[1] - self.byte_range[0] + 1
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers cancel an old stream when the playlist changes clips.
            pass

    def do_GET(self):
        if unquote(urlsplit(self.path).path) not in PAGES | VIDEOS:
            self.send_error(404)
            return
        if urlsplit(self.path).path == '/':
            self.send_response(302)
            self.send_header('Location', '/docs/demo/index.html')
            self.end_headers()
            return
        super().do_GET()

    def do_HEAD(self):
        if unquote(urlsplit(self.path).path) not in PAGES | VIDEOS:
            self.send_error(404)
            return
        if urlsplit(self.path).path == '/':
            self.send_response(302)
            self.send_header('Location', '/docs/demo/index.html')
            self.end_headers()
            return
        super().do_HEAD()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    missing = [path for path in VIDEOS if not (ROOT / path.lstrip('/')).is_file()]
    if missing:
        parser.error('Approved demo videos are missing: ' + ', '.join(missing))
    handler = partial(DemoHandler, directory=str(ROOT))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Demo: http://localhost:{args.port}/ (Ctrl+C to stop)', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
