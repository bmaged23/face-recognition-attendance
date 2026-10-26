"""Publish measured research reports and a small dependency-free SVG chart."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/evaluation/measurements'
NAMES = ['native-cpu', 'native-cuda', 'native-cuda-compact', 'dev-efficient-detector', 'dev-threshold-080']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        shutil.copy2(ROOT / f'output/evaluation/{name}.json', OUT / f'{name}.json')
    cpu = json.loads((OUT / 'native-cpu.json').read_text())
    gpu = json.loads((OUT / 'native-cuda.json').read_text())
    compact = json.loads((OUT / 'native-cuda-compact.json').read_text())
    full_outcomes = [(r['case_id'], r['detected'], r['prediction']) for r in gpu['samples']]
    assert full_outcomes == [(r['case_id'], r['detected'], r['prediction']) for r in cpu['samples']]
    assert full_outcomes == [(r['case_id'], r['detected'], r['prediction']) for r in compact['samples']]
    reference = {'version': 1, 'kind': 'Frozen model regression baseline, NOT human ground truth',
                 'human_reference': 'cases.json identities and eye annotations from ChokePoint',
                 'fixture_sha256': compact['fixture_sha256'], 'model_signature': compact['model_signature'],
                 'required_gallery_count': 19, 'required_case_count': 228,
                 'min_correct_known': 113, 'min_detected_unknown': 56,
                 'max_wrong_known': 0, 'max_unknown_false_accepts': 0,
                 'outcomes': [{'case_id': r['case_id'], 'detected': r['detected'], 'prediction': r['prediction']}
                              for r in compact['samples']]}
    (ROOT / 'evaluation/reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    shutil.copy2(ROOT / 'input/recognition-evaluation/manifest.json', ROOT / 'evaluation/cases.json')
    export = json.loads((ROOT / 'vendor/adaface/pretrained/adaface_ir100_webface12m.backbone.json').read_text())
    (OUT / 'checkpoint-export.json').write_text(json.dumps(export, indent=2) + '\n')
    c = cpu['overall']['latency_seconds']['median'] * 1000
    g = gpu['overall']['latency_seconds']['median'] * 1000
    rows = [('CPU probe', c, 1400, '#8fb0d5', f'{c:.1f} ms'),
            ('GPU probe', g, 1400, '#64dec7', f'{g:.1f} ms'),
            ('Original weights', export['source_bytes']/1e6, 1600, '#8fb0d5', f"{export['source_bytes']/1e6:.1f} MB"),
            ('Inference weights', export['export_bytes']/1e6, 1600, '#64dec7', f"{export['export_bytes']/1e6:.1f} MB"),
            ('Original peak RAM', gpu['resources']['rss_mib']['peak'], 2300, '#8fb0d5', f"{gpu['resources']['rss_mib']['peak']:.0f} MiB"),
            ('Compact peak RAM', compact['resources']['rss_mib']['peak'], 2300, '#64dec7', f"{compact['resources']['rss_mib']['peak']:.0f} MiB")]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 570" role="img" aria-labelledby="title desc">',
         '<title id="title">Measured recognition latency and checkpoint optimization</title>',
         '<desc id="desc">Matched 228 CPU and GPU probe outcomes. Inference-only export reduces checkpoint size and sampled startup/run RAM.</desc>',
         '<rect width="1000" height="570" rx="18" fill="#0b1220"/>',
         '<g font-family="Arial,sans-serif" fill="#e7edf7">',
         '<text x="30" y="42" font-size="24" font-weight="bold">Measured native recognition performance</text>',
         '<text x="30" y="70" font-size="15" fill="#abbad0">GTX 1660 Ti · two inference threads · 228 fixed probes · CUDA model, local Qdrant</text>']
    for i,(label,value,scale,color,value_label) in enumerate(rows):
        y=110+i*58
        svg += [f'<text x="30" y="{y+22}" font-size="17">{label}</text>',
                f'<rect x="240" y="{y}" width="{max(3,value/scale*540):.1f}" height="30" rx="5" fill="{color}"/>',
                f'<text x="805" y="{y+22}" font-size="17">{value_label}</text>']
    svg += [f'<text x="30" y="495" font-size="19" fill="#64dec7">{c/g:.2f}× lower median probe latency; identical CPU/GPU identity outcomes.</text>',
            '<text x="30" y="523" font-size="15" fill="#abbad0">Probe timing excludes gallery enrollment, startup and video encoding. RAM includes sampled startup.</text>',
            '<text x="30" y="548" font-size="15" fill="#abbad0">113/152 known probes recognized overall; all 113 detected known probes correct. Small research set.</text>',
            '</g></svg>']
    (ROOT / 'docs/evaluation/native-performance.svg').write_text('\n'.join(svg)+'\n')
    print(f'Published five native reports; matched probe median ratio {c/g:.2f}x')


if __name__ == '__main__':
    main()
