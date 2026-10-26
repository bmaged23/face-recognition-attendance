"""Test the exact candidate container with real GPU models and isolated data."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import httpx

ROOT = Path(__file__).resolve().parents[2]


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/container-gate')
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'input/recognition-evaluation')
    parser.add_argument('--public-smoke', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    name = 'face-public-verification' if args.public_smoke else 'face-qualification'
    actual_sha = run('docker', 'image', 'inspect', args.image, '--format',
                     '{{index .Config.Labels "org.opencontainers.image.revision"}}', capture_output=True).stdout.strip()
    if actual_sha != args.source_sha:
        raise RuntimeError('Candidate source label does not match this release')
    # This folder is disposable and unique per attempt. A non-empty folder must
    # never silently reuse gallery entries or synthetic attendance from a prior run.
    data = args.output / 'container-data'
    data.mkdir(exist_ok=False)
    command = ['docker','run','-d','--rm','--gpus','all','--name',name,
               '-p','127.0.0.1::7323',
               '-v',f'{data}:/app/data', '-v',f'{args.fixtures.resolve()}:/fixtures:ro',
               '-v',f'{ROOT / "evaluation"}:/gate:ro', args.image]
    run(*command)
    results = {'image': args.image, 'source_sha': args.source_sha, 'status': 'running'}
    try:
        # Tests execute code and frozen dependencies shipped inside the image.
        if not args.public_smoke:
            run('docker','exec',name,'python','-m','unittest','discover','-s','tests','-v')
            run('docker','exec',name,'python','-m','pip','check')
            run('docker','exec',name,'python','scripts/evaluation/evaluate_recognition.py',
                '--device','cuda','--cases','/fixtures',
                '--model-path','/app/vendor/adaface/pretrained/adaface_ir100_webface12m.backbone.pt',
                '--output','/app/data/recognition.json')
            run('docker','exec',name,'python','scripts/release/assert_quality.py',
                '--report','/app/data/recognition.json','--reference','/gate/reference.json')
            report = json.loads((data / 'recognition.json').read_text())
            results['recognition'] = report['overall']
            results['resources'] = report['resources']
        # Retry server listening only; inference failures below fail the gate.
        for _ in range(30):
            probe=subprocess.run(['docker','exec',name,'python','-c',
                 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:7323/health",timeout=2)'],capture_output=True)
            if probe.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Container HTTP server did not start')
        published = run('docker','port',name,'7323/tcp',capture_output=True).stdout.strip()
        response = httpx.get('http://' + published + '/health',timeout=10,trust_env=False)
        response.raise_for_status()
        results['published_host_http'] = 'passed'
        run('docker','exec',name,'python','scripts/validation/validate_real_api.py',
            '--cases','/fixtures','--workdir','/app/data/http-smoke')
        smoke = json.loads((data / 'http-smoke/api-smoke.json').read_text())
        if smoke['readiness']['device'] != 'cuda' or smoke['readiness']['storage'] != 'embedded':
            raise RuntimeError('Container did not use the required GPU/embedded store')
        results['http_checks'] = smoke['checks']
        results['http_events'] = smoke['events']
        # Cold restart of the exact image and volume must preserve enrolled people.
        run('docker','stop',name)
        run(*command)
        identity, person = smoke['enrolled_people'][0]
        test = f"""import json,time,httpx
from pathlib import Path
with httpx.Client(base_url='http://127.0.0.1:7323',timeout=120,trust_env=False) as client:
    for attempt in range(30):
        try:
            client.get('/health').raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(1)
    response=client.get('/ready'); response.raise_for_status()
    assert response.json()['device']=='cuda'
    manifest=json.loads(Path('/fixtures/manifest.json').read_text())
    row=next(r for r in manifest['cases'] if r['identity']=={identity!r} and r['camera']==2)
    response=client.post('/recognize/image',files={{'image':('probe.jpg',(Path('/fixtures')/row['source']).read_bytes(),'image/jpeg')}})
    response.raise_for_status()
    assert {person['id']!r} in {{r['id'] for r in response.json()}}
print('Real HTTP recognition after persistent gallery restart passed')
"""
        run('docker','exec',name,'python','-c',test)
        results['persistent_gallery_restart'] = 'passed'
        results['status'] = 'passed'
    except Exception:
        results['status'] = 'failed'
        raise
    finally:
        logs=subprocess.run(['docker','logs',name],capture_output=True,text=True)
        (args.output/'container.log').write_text(logs.stdout + logs.stderr)
        subprocess.run(['docker','stop',name],capture_output=True)
        (args.output/'gate-summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
