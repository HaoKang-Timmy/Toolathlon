#!/usr/bin/env python3
"""Run the cache-cold Toolathlon 16/32/48/64 suite and retain all trajectories."""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from deployment import deploy, healthy, load_config


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def flush_cache(cfg: dict) -> dict:
    results = {}
    for worker in cfg["workers"]:
        url = f"http://{worker['ip']}:{worker['port']}/flush_cache"
        error = None
        for method in ("POST", "GET"):
            try:
                request = urllib.request.Request(url, method=method)
                with urllib.request.urlopen(request, timeout=60) as response:
                    results[worker["name"]] = {"method": method, "status": response.status, "body": response.read().decode(errors="replace")}
                    break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        else:
            raise RuntimeError(f"cache flush failed for {worker['name']}: {error}")
    return results


def capture_manifest(run_dir: Path, cfg: dict) -> None:
    manifest = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "config": cfg, "commands": {}}
    commands = {
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
        "docker_images": ["docker", "images", "--digests"],
        "nvidia_smi": ["nvidia-smi"],
        "disk": ["df", "-h", "/", "/mnt/shared"],
        "python_packages": [cfg["python"], "-m", "uv", "pip", "freeze"],
    }
    for name, command in commands.items():
        try:
            result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=120)
            manifest["commands"][name] = {"argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as exc:
            manifest["commands"][name] = {"argv": command, "error": f"{type(exc).__name__}: {exc}"}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def expected_tasks(cfg: dict) -> int:
    override = os.environ.get("TASK_LIST")
    if override and Path(override).exists():
        return sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in Path(override).read_text().splitlines())
    return sum(path.is_dir() for path in (REPO / "tasks" / cfg["task_folder"]).iterdir())


def completed_tasks(level_dir: Path) -> int:
    return len(list(level_dir.glob("finalpool/*/traj_log.json")))


def retry_task_list(level_dir: Path, cfg: dict) -> Path | None:
    override = os.environ.get("TASK_LIST")
    if override and Path(override).exists():
        all_tasks = {line.strip() for line in Path(override).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")}
    else:
        all_tasks = {path.name for path in (REPO / "tasks" / cfg["task_folder"]).iterdir() if path.is_dir()}
    done = {path.parent.name for path in level_dir.glob("finalpool/*/traj_log.json")}
    missing = sorted(all_tasks - done)
    if not missing:
        return None
    path = level_dir / "missing_tasks.txt"
    path.write_text("".join(f"{name}\n" for name in missing))
    return path


def run_level(cfg: dict, run_dir: Path, concurrency: int, max_repair_attempts: int) -> None:
    level_dir = run_dir / f"concurrency-{concurrency}"
    level_dir.mkdir(parents=True, exist_ok=True)
    if (level_dir / "complete.marker").exists():
        print(f"already complete: concurrency={concurrency}", flush=True)
        return
    cache_record = flush_cache(cfg)
    (level_dir / "cache_flush.json").write_text(json.dumps(cache_record, indent=2) + "\n")
    env = os.environ.copy()
    original_task_list = env.get("TASK_LIST")
    env.update({
        "TOOLATHLON_OPENAI_BASE_URL": f"http://{cfg['router']['host']}:{cfg['router']['port']}/v1",
        "TOOLATHLON_OPENAI_API_KEY": "toolathlon-local",
        "TOOLATHLON_CAPTURE_REQUEST_METRICS": "1",
        "TOOLATHLON_AGENT_FRAMEWORK": "toolathlon_default",
    })
    attempt = 0
    while completed_tasks(level_dir) < expected_tasks(cfg) and attempt < max_repair_attempts:
        attempt += 1
        missing = retry_task_list(level_dir, cfg) if attempt > 1 else None
        if missing:
            env["TASK_LIST"] = str(missing)
        elif original_task_list:
            env["TASK_LIST"] = original_task_list
        else:
            env.pop("TASK_LIST", None)
        command = [
            "bash", "scripts/run_parallel.sh", cfg["served_model_name"], str(level_dir), "unified",
            str(concurrency), cfg["docker_image"], "", "decoupled", "normal", "toolathlon_default",
        ]
        log = level_dir / f"suite-attempt-{attempt}.log"
        with log.open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        count = completed_tasks(level_dir)
        print(f"concurrency={concurrency} attempt={attempt} exit={result.returncode} trajectories={count}/{expected_tasks(cfg)}", flush=True)
        if count < expected_tasks(cfg):
            # Infrastructure and task failures are diagnosed in retained logs; resume only missing tasks.
            time.sleep(min(60, 10 * attempt))
    if completed_tasks(level_dir) != expected_tasks(cfg):
        raise RuntimeError(f"concurrency={concurrency} incomplete after repair retries: {completed_tasks(level_dir)}/{expected_tasks(cfg)}")
    (level_dir / "complete.marker").write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run-id", default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--reuse-cluster", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one task at concurrency 1 instead of the full suite")
    parser.add_argument("--max-repair-attempts", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(cfg["results_root"]) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    capture_manifest(run_dir, cfg)
    if not args.reuse_cluster:
        deploy(cfg, run_dir, 3600)
    elif not healthy(f"http://{cfg['router']['host']}:{cfg['router']['port']}"):
        raise RuntimeError("--reuse-cluster requested but router is not healthy")

    stop_file = run_dir / "monitor.stop"
    stop_file.unlink(missing_ok=True)
    monitor = subprocess.Popen([
        sys.executable, str(HERE / "monitor.py"), "--output", str(run_dir / "monitor"), "--stop-file", str(stop_file),
        "--interval", str(cfg["monitor_interval_seconds"]),
    ], cwd=HERE, stdout=(run_dir / "monitor.log").open("a"), stderr=subprocess.STDOUT)
    try:
        if args.smoke:
            smoke_cfg = dict(cfg)
            smoke_cfg["concurrency"] = [1]
            task = next(path.name for path in (REPO / "tasks" / cfg["task_folder"]).iterdir() if path.is_dir())
            task_list = run_dir / "smoke_task.txt"
            task_list.write_text(task + "\n")
            os.environ["TASK_LIST"] = str(task_list)
            run_level(smoke_cfg, run_dir, 1, args.max_repair_attempts)
        else:
            for concurrency in cfg["concurrency"]:
                run_level(cfg, run_dir, concurrency, args.max_repair_attempts)
    finally:
        stop_file.touch()
        try:
            monitor.wait(timeout=90)
        except subprocess.TimeoutExpired:
            monitor.terminate()
        subprocess.run([sys.executable, str(HERE / "report.py"), str(run_dir)], cwd=REPO, check=False)
    print(run_dir)


if __name__ == "__main__":
    main()
