"""Sample Docker CPU/memory and whole-device VRAM; no payload or credential capture."""

import argparse
import json
from pathlib import Path
import re
import signal
import subprocess
import time


def memory_mib(value):
    match = re.match(r"([\d.]+)\s*([A-Za-z]+)", value)
    if not match:
        raise ValueError("Unrecognized Docker memory unit")
    amount, unit = match.groups()
    return (
        float(amount)
        * {
            "B": 1 / 1048576,
            "kB": 1000 / 1048576,
            "KiB": 1 / 1024,
            "MB": 1000000 / 1048576,
            "MiB": 1,
            "GB": 1000000000 / 1048576,
            "GiB": 1024,
        }[unit]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="face-qualification")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=int, default=1800)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def finish(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, finish)
    signal.signal(signal.SIGINT, finish)
    samples = []
    started = time.monotonic()
    while not stop and time.monotonic() - started < args.maximum_seconds:
        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{json .}}",
                args.container,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            raw = json.loads(result.stdout)
            row = {
                "elapsed_seconds": time.monotonic() - started,
                "cpu_percent_one_core": float(raw["CPUPerc"].strip("%")),
                "memory_mib": memory_mib(raw["MemUsage"].split("/")[0]),
                "pids": int(raw["PIDs"]),
            }
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if gpu.returncode == 0:
                values = gpu.stdout.splitlines()[0].split(",")
                row.update(
                    gpu_vram_mib=float(values[0]), gpu_util_percent=float(values[1])
                )
            samples.append(row)
            summary = {
                "samples": len(samples),
                "metrics": {
                    key: {
                        "mean": sum(r[key] for r in samples if key in r)
                        / sum(key in r for r in samples),
                        "peak": max(r[key] for r in samples if key in r),
                    }
                    for key in row
                    if key != "elapsed_seconds"
                },
                "scope": "Docker cgroup CPU/memory including API, vision workers and FFmpeg; CPU100%=one logical core. GPU is whole-device usage. No cloud inference is used. Coarse sampling can miss peaks.",
            }
            args.output.write_text(json.dumps(summary, indent=2) + "\n")
        time.sleep(1)
    print(json.dumps({"recorded_samples": len(samples), "output": str(args.output)}))


if __name__ == "__main__":
    main()
