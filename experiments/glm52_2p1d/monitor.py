#!/usr/bin/env python3
"""Persist five-minute GPU and SGLang metric snapshots for all three nodes."""

import argparse
import csv
import datetime as dt
import io
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from deployment import load_config


GPU_FIELDS = "index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"


def gpu_sample(worker: dict) -> list[dict]:
    command = ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"]
    if worker["ssh"]:
        command = ["ssh", "-o", "BatchMode=yes", worker["ssh"], *command]
    output = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30).stdout
    rows = []
    for values in csv.reader(io.StringIO(output), skipinitialspace=True):
        rows.append(dict(zip(GPU_FIELDS.split(","), values)))
    return rows


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read().decode("utf-8", errors="replace")


def selected_prometheus(text: str) -> list[dict]:
    selected = []
    names = ("cache_hit_rate", "token_usage", "num_used_tokens", "pending_prealloc_token_usage", "prompt_tokens_total", "generation_tokens_total")
    for line in text.splitlines():
        if not line or line.startswith("#") or not line.startswith("sglang:"):
            continue
        metric, _, value = line.rpartition(" ")
        if any(name in metric for name in names):
            try:
                selected.append({"metric": metric, "value": float(value)})
            except ValueError:
                pass
    return selected


def take_sample(cfg: dict, output: Path, sequence: int) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    sample_dir = output / "prometheus" / f"{sequence:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": stamp, "sequence": sequence, "gpus": {}, "sglang": {}, "errors": []}
    for worker in cfg["workers"]:
        try:
            record["gpus"][worker["name"]] = gpu_sample(worker)
        except Exception as exc:
            record["errors"].append(f"gpu {worker['name']}: {type(exc).__name__}: {exc}")
        try:
            raw = fetch_text(f"http://{worker['ip']}:{worker['port']}/metrics")
            (sample_dir / f"{worker['name']}.prom").write_text(raw)
            record["sglang"][worker["name"]] = selected_prometheus(raw)
        except Exception as exc:
            record["errors"].append(f"metrics {worker['name']}: {type(exc).__name__}: {exc}")
    with (output / "samples.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--interval", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output, stop_file = Path(args.output), Path(args.stop_file)
    output.mkdir(parents=True, exist_ok=True)
    interval = args.interval or cfg["monitor_interval_seconds"]
    sequence = 0
    while True:
        take_sample(cfg, output, sequence)
        sequence += 1
        if stop_file.exists():
            break
        deadline = time.monotonic() + interval
        while time.monotonic() < deadline and not stop_file.exists():
            time.sleep(min(5, deadline - time.monotonic()))
        if stop_file.exists():
            take_sample(cfg, output, sequence)
            break


if __name__ == "__main__":
    main()
