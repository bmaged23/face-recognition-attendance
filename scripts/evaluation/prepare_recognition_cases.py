"""Freeze labeled ChokePoint cases without selecting for model success.

Run from the project root after obtaining the three official archives. No learned
model is used in sampling. Raw images stay under ignored input/.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'input/chokepoint'
DEST = ROOT / 'input/recognition-evaluation'
ARCHIVES = {
    'Still.tar.xz': '2efcd4010d19e811cce144a614ea82a2',
    'groundtruth.tar.xz': '5c9d4d38a1c614905fe48da48ecdf06c',
    'P2E_S2.tar.xz': 'cc464106ef29e363b67fdfbfd5e25ed9',
}


def digest(path, algorithm='sha256'):
    with path.open('rb') as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def main():
    for name, checksum in ARCHIVES.items():
        assert digest(DATA / name, 'md5') == checksum, f'Invalid archive: {name}'
    DEST.mkdir(parents=True, exist_ok=True)
    cases = []
    with tarfile.open(DATA / 'groundtruth.tar.xz') as archive:
        for camera in (2, 1):
            candidates = {}
            for member in sorted(archive.getmembers(), key=lambda m: m.name):
                if f'P2E_S2_C{camera}.' not in member.name or not member.name.endswith('.xml'):
                    continue
                root = ET.fromstring(archive.extractfile(member).read())
                for frame in root.findall('frame'):
                    for person in frame.findall('person'):
                        eyes = [person.find(name) for name in ('leftEye', 'rightEye')]
                        if any(eye is None for eye in eyes):
                            continue
                        identity = person.attrib['id']
                        source = f"{Path(member.name).stem}/{frame.attrib['number']}.jpg"
                        row = {'identity': identity, 'camera': camera, 'source': source,
                               'eyes': [[float(e.attrib['x']), float(e.attrib['y'])] for e in eyes]}
                        candidates.setdefault(identity, {})[source] = row
            for identity in sorted(candidates):
                rows = [row for _, row in sorted(candidates[identity].items())]
                count = min(4, len(rows))
                indices = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
                for i in indices:
                    row = dict(rows[i])
                    row.update(case_id=f"{Path(row['source']).parent.name}-{identity}-{Path(row['source']).stem}",
                               known=int(identity) % 3 != 0,
                               split='development' if camera == 2 else 'held-out-view')
                    cases.append(row)
    required = {row['source'] for row in cases}
    with tarfile.open(DATA / 'P2E_S2.tar.xz') as outer:
        for camera in (1, 2):
            name = f'P2E_S2_C{camera}.tar.xz'
            with tarfile.open(fileobj=outer.extractfile(name), mode='r|xz') as inner:
                for member in inner:
                    name = member.name.removeprefix('./')
                    if name not in required or not member.isfile():
                        continue
                    target = DEST / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with inner.extractfile(member) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
    identities = sorted({row['identity'] for row in cases if row['known']})
    gallery = []
    with tarfile.open(DATA / 'Still.tar.xz') as archive:
        for identity in identities:
            matches = [m for m in archive.getmembers() if m.name.endswith(f'Neutral/ID{identity}.JPG')]
            assert len(matches) == 1, (identity, [m.name for m in matches])
            target = DEST / 'gallery' / f'{identity}.JPG'
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(matches[0]) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            gallery.append({'identity': identity, 'path': target.relative_to(DEST).as_posix(),
                            'sha256': digest(target)})
    for row in cases:
        row['sha256'] = digest(DEST / row['source'])
    manifest = {'version': 1, 'dataset': 'ChokePoint P2E S2', 'archives_md5': ARCHIVES,
                'sampling': '4 equally spaced annotated frames per identity/view; neutral still only',
                'unknown_split': 'identity integer divisible by 3; no unknown identity enrolled',
                'limitations': 'Camera views share the same recording. Repeated frames are correlated. Not an independent-session or demographic benchmark.',
                'gallery': gallery, 'cases': cases}
    encoded = json.dumps(manifest, indent=2) + '\n'
    (DEST / 'manifest.json').write_text(encoded)
    print(json.dumps({'cases': len(cases), 'gallery': len(gallery),
                      'unknown_identities': sorted({r['identity'] for r in cases if not r['known']}),
                      'manifest_sha256': hashlib.sha256(encoded.encode()).hexdigest()}))


if __name__ == '__main__':
    main()
