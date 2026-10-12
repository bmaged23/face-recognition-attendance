"""Download the tested AdaFace checkpoint atomically and verify its SHA-256."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'vendor/adaface/pretrained/adaface_ir100_webface12m.ckpt'
URL = ('https://huggingface.co/VishalMishraTss/AdaFace/resolve/'
       '534fa44f9499437645bd18eee376e52fd60bd931/adaface_ir101_webface12m.ckpt')
SHA256 = '0e7a3238d2a50f3fe3860782534928ac7cb2598977cf897f6869fd5ac2493fd0'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DEFAULT)
    parser.add_argument('--verify', action='store_true', help='Verify an existing file without downloading')
    args = parser.parse_args()
    destination = args.destination
    if destination.is_file():
        if digest(destination) == SHA256:
            print(f'Checkpoint verified: {destination}')
            return
        raise SystemExit(f'Checksum mismatch: {destination}. Move the incorrect file before downloading.')
    if args.verify:
        raise SystemExit(f'Checkpoint not found: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.download')
    try:
        print('Downloading the pinned WebFace12M checkpoint (~1.53 GB).', flush=True)
        request = urllib.request.Request(URL, headers={'User-Agent': 'FaceRecognitionSetup/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            downloaded = 0
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded % (128 * 1024 * 1024) == 0:
                    print(f'{downloaded / 1024**2:.0f} MiB downloaded', flush=True)
        if digest(temporary) != SHA256:
            raise RuntimeError('Downloaded checkpoint failed SHA-256 verification')
        temporary.replace(destination)
        print(f'Checkpoint downloaded and verified: {destination}')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
