"""Drop unused AdaFace training tensors, retaining exact inference parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.download_model import DEFAULT, SHA256, digest
from face_recognition.core.backend import _checkpoint_compatibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--destination', type=Path, default=DEFAULT.with_suffix('.backbone.pt'))
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise ValueError('Keep the verified source checkpoint separate from the export')
    if digest(args.source) != SHA256:
        raise ValueError('Source checkpoint failed the frozen upstream SHA256 check')
    import torch
    _checkpoint_compatibility()
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
    raw = checkpoint['state_dict']
    backbone = {name.removeprefix('model.'): tensor for name, tensor in raw.items() if name.startswith('model.')}
    if not backbone:
        raise ValueError('No inference parameters in checkpoint')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone, args.destination)
    restored = torch.load(args.destination, map_location='cpu', weights_only=True)
    if set(backbone) != set(restored) or not all(torch.equal(tensor, restored[key]) for key, tensor in backbone.items()):
        raise RuntimeError('Export failed exact tensor equality')
    report = {'source_sha256': SHA256, 'export_sha256': digest(args.destination),
              'source_bytes': args.source.stat().st_size, 'export_bytes': args.destination.stat().st_size,
              'tensor_count': len(backbone), 'exact_tensor_equality': True,
              'note': 'File fingerprints differ. Re-enroll galleries when selecting this export.'}
    args.destination.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
