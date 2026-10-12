"""Create full-frame camera-2 test MP4s from original crowded ChokePoint data.

The resulting videos are modified derivatives of NICTA's ChokePoint dataset,
for noncommercial experimentation. Retain input/chokepoint/SOURCE_AND_LICENSE.html.
"""
import hashlib
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from face_recognition.video.encoding import finalize_video

CHECKSUMS = {
    'P2E_S5': 'e6bca312e40ebebdfb709f2315eebf80',
    'P2L_S5': '14b28f3d83a75b62d3aa6ad5ee117843',
}


def main():
    root = ROOT / 'input' / 'chokepoint'
    for sequence, expected in CHECKSUMS.items():
        archive = root / f'{sequence}.tar.xz'
        with archive.open('rb') as file:
            assert hashlib.file_digest(file, 'md5').hexdigest() == expected, f'Invalid download: {archive}'
        with tarfile.open(archive) as outer:
            member = next(m for m in outer if m.name == f'{sequence}_C2.tar.xz')
            with tarfile.open(fileobj=outer.extractfile(member), mode='r|xz') as inner:
                inner.extractall(root / 'frames', filter='data')
        folder = root / 'frames' / f'{sequence}_C2.1'
        frames = {int(p.stem): p for p in folder.glob('*.jpg')}
        path = root / f'{sequence}_C2.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (800, 600))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot create {path}')
        last = None
        missing = 0
        try:
            for number in range(min(frames), max(frames) + 1):
                if number in frames:
                    last = cv2.imread(str(frames[number]))
                    if last is None:
                        raise RuntimeError(f'Unreadable frame: {frames[number]}')
                else:
                    missing += 1
                # Hold the previous frame across source gaps to retain source timing.
                writer.write(last)
        finally:
            writer.release()
        finalize_video(path)
        print(f'{path}: {len(frames)} source frames, {missing} missing frames held, 30 fps', flush=True)
    (root / 'MODIFIED_DATA_NOTICE.txt').write_text(
        'MODIFIED DERIVATIVES OF THE NICTA CHOKEPOINT DATASET\n'
        'These MP4 files were generated from original camera-2 JPEG frames at 30 fps.\n'
        'Missing source frames repeat the previous frame. Audio is absent.\n'
        'Annotated outputs additionally contain automatic face boxes and test aliases.\n'
        'Original frames and archives are retained. Noncommercial experimentation only.\n'
        'Retain SOURCE_AND_LICENSE.html with copies and derivatives.\n'
        'Source: https://arma.sourceforge.net/chokepoint/\n'
        'NICTA; Y. Wong, S. Chen, S. Mau, C. Sanderson, B.C. Lovell,\n'
        'Patch-based Probabilistic Image Quality Assessment for Face Selection and\n'
        'Improved Video-based Face Recognition, CVPR Workshops, pp. 81-88, 2011.\n'
    )


if __name__ == '__main__':
    main()
