"""Finalize OpenCV recordings as broadly playable H.264 MP4 files."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def require_ffmpeg():
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('Annotated video output requires FFmpeg with the libx264 encoder; install ffmpeg first')
    return executable


def finalize_video(path):
    path = Path(path)
    executable = require_ffmpeg()
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.mp4', delete=False) as file:
        temporary = Path(file.name)
    try:
        result = subprocess.run(
            [executable, '-hide_banner', '-loglevel', 'error', '-y',
             '-i', str(path.resolve()), '-map', '0:v:0', '-an',
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
             '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', str(temporary.resolve())],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f'Cannot encode annotated video as H.264: {result.stderr.strip()}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
