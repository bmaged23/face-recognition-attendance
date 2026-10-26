"""Summarize executed stages only; skipped model calls are not zero-ms inference."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def main():
    reports = ROOT / 'docs/evaluation/measurements'
    summary = {'scope': 'Native warmed batch-one stages from the matched 228-probe study; excludes startup/enrollment/video encoding. Embedding includes tensor preparation and transfer. Search is local in-memory Qdrant.'}
    for label in ('cpu', 'cuda'):
        report = json.loads((reports / f'native-{label}.json').read_text())
        stages = {}
        for name, detected_only in [('detection_seconds', False), ('embedding_seconds', True), ('search_seconds', True)]:
            values = [row[name] * 1000 for row in report['samples'] if not detected_only or row['detected']]
            stages[name.removesuffix('_seconds')] = {'executions': len(values), 'median_ms': statistics.median(values)}
        summary[label] = stages
    (reports / 'component-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
