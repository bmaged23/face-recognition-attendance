"""Check exact-image recognition against human labels and frozen regressions."""
import argparse
import json
from pathlib import Path


def check(report, reference):
    errors = []
    if report.get('fixture_sha256') != reference['fixture_sha256']:
        errors.append('Fixture fingerprint differs')
    if report.get('model_signature') != reference['model_signature']:
        errors.append('Model fingerprint differs')
    if report.get('device') != 'cuda':
        errors.append('Required CUDA execution was not used')
    if len(report.get('samples', [])) != reference['required_case_count']:
        errors.append('Incomplete probe execution')
    if sum(row['enrolled'] for row in report.get('gallery', [])) != reference['required_gallery_count']:
        errors.append('Incomplete gallery enrollment')
    actual = {row['case_id']: (row['detected'], row['prediction']) for row in report.get('samples', [])}
    expected = {row['case_id']: (row['detected'], row['prediction']) for row in reference['outcomes']}
    if actual != expected:
        errors.append('At least one frozen detection or identity outcome changed; review required')
    metrics = report.get('overall', {})
    for name, limit, mode in [
        ('correct_known_accepted', reference['min_correct_known'], 'min'),
        ('detected_unknown', reference['min_detected_unknown'], 'min'),
        ('wrong_known_accepted', reference['max_wrong_known'], 'max'),
        ('unknown_false_accepts', reference['max_unknown_false_accepts'], 'max'),
    ]:
        value = metrics.get(name)
        if value is None or (value < limit if mode == 'min' else value > limit):
            errors.append(f'Quality threshold failed: {name}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    errors = check(json.loads(args.report.read_text()), json.loads(args.reference.read_text()))
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Exact-image recognition quality gate passed.')


if __name__ == '__main__':
    main()
