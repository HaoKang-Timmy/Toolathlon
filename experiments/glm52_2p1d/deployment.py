#!/usr/bin/env python3
"""Deploy/stop the fixed three-node GLM-5.2 2P1D SGLang cluster."""

import argparse
import json
import os
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_config(path: str | None) -> dict:
    return json.loads(Path(path or HERE / "cluster.json").read_text())


def run_on(host: str | None, command: str, check: bool = True) -> subprocess.CompletedProcess:
    argv = ["bash", "-lc", command] if host is None else ["ssh", "-o", "BatchMode=yes", host, "bash", "-lc", shlex.quote(command)]
    return subprocess.run(argv, text=True, check=check, capture_output=True)


def common_args(cfg: dict) -> list[str]:
    return [
        "--model-path", cfg["model_path"],
        "--served-model-name", cfg["served_model_name"],
        "--trust-remote-code",
        "--quantization", "fp8",
        "--tp-size", "8",
        "--attn-cp-size", "8",
        "--moe-dp-size", "1",
        "--ep-size", "8",
        "--moe-a2a-backend", "none",
        "--moe-runner-backend", "flashinfer_trtllm",
        "--attention-backend", "dsa",
        "--dsa-prefill-backend", "flashmla_sparse",
        "--dsa-decode-backend", "flashmla_kv",
        "--dsa-topk-backend", "sgl-kernel",
        "--kv-cache-dtype", "fp8_e4m3",
        "--context-length", "262144",
        "--chunked-prefill-size", "32768",
        "--mem-fraction-static", "0.85",
        "--reasoning-parser", "glm45",
        "--tool-call-parser", "glm45",
        "--disable-shared-experts-fusion",
        "--disable-custom-all-reduce",
        "--enable-metrics",
        "--enable-metrics-for-all-schedulers",
        "--enable-cache-report",
        "--stream-response-default-include-usage",
        "--speculative-algorithm", "EAGLE",
        "--speculative-draft-model-path", cfg["model_path"],
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
        "--speculative-moe-runner-backend", "flashinfer_trtllm",
    ]


def worker_args(cfg: dict, worker: dict) -> list[str]:
    args = common_args(cfg) + [
        "--host", "0.0.0.0",
        "--port", str(worker["port"]),
        "--disaggregation-mode", worker["role"],
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(worker["bootstrap_port"]),
        "--disaggregation-ib-device", json.dumps(cfg["ib_device_map"], separators=(",", ":")),
    ]
    if worker["role"] == "prefill":
        args += ["--enable-prefill-cp", "--cp-strategy", "zigzag", "--enable-dsa-cp-shared-kv-cache"]
    else:
        # GLM DSA explicitly forbids --enable-prefill-cp on a PD decode worker.
        args += ["--enable-dsa-cp-shared-kv-cache"]
    return args


def launch_worker(cfg: dict, worker: dict, run_dir: Path) -> None:
    log = run_dir / "servers" / f"{worker['name']}.log"
    pid = run_dir / "pids" / f"{worker['name']}.pid"
    args = " ".join(shlex.quote(x) for x in worker_args(cfg, worker))
    command = (
        f"mkdir -p {shlex.quote(str(log.parent))} {shlex.quote(str(pid.parent))}; "
        f"test ! -f {shlex.quote(str(pid))} || kill $(cat {shlex.quote(str(pid))}) 2>/dev/null || true; "
        f"nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
        f"PYTHONPATH={shlex.quote(cfg['pythonpath'])} "
        f"{shlex.quote(cfg['python'])} -m sglang.launch_server {args} "
        f"> {shlex.quote(str(log))} 2>&1 < /dev/null & echo $! > {shlex.quote(str(pid))}"
    )
    run_on(worker["ssh"], command)


def healthy(url: str, timeout: float = 3) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def wait_workers(cfg: dict, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    pending = {w["name"]: f"http://{w['ip']}:{w['port']}" for w in cfg["workers"]}
    while pending and time.monotonic() < deadline:
        for name, url in list(pending.items()):
            if healthy(url):
                print(f"healthy: {name} {url}", flush=True)
                pending.pop(name)
        if pending:
            time.sleep(10)
    if pending:
        raise TimeoutError(f"workers did not become healthy: {pending}")


def launch_router(cfg: dict, run_dir: Path) -> None:
    router = cfg["router"]
    log = run_dir / "servers" / "router.log"
    pid = run_dir / "pids" / "router.pid"
    args = ["--host", router["host"], "--port", str(router["port"]), "--policy", "cache_aware"]
    for worker in cfg["workers"]:
        url = f"http://{worker['ip']}:{worker['port']}"
        args += ["--prefill", url, str(worker["bootstrap_port"])] if worker["role"] == "prefill" else ["--decode", url]
    args += ["--prometheus-host", "127.0.0.1", "--prometheus-port", str(router["prometheus_port"])]
    command = (
        f"test ! -f {shlex.quote(str(pid))} || kill $(cat {shlex.quote(str(pid))}) 2>/dev/null || true; "
        f"nohup env PYTHONPATH={shlex.quote(cfg['pythonpath'])} {shlex.quote(cfg['python'])} "
        f"-m sglang_router.launch_router {' '.join(shlex.quote(x) for x in args)} "
        f"> {shlex.quote(str(log))} 2>&1 < /dev/null & echo $! > {shlex.quote(str(pid))}"
    )
    run_on(None, command)


def deploy(cfg: dict, run_dir: Path, timeout: int) -> None:
    (run_dir / "servers").mkdir(parents=True, exist_ok=True)
    (run_dir / "pids").mkdir(parents=True, exist_ok=True)
    (run_dir / "cluster.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for worker in cfg["workers"]:
        launch_worker(cfg, worker, run_dir)
    wait_workers(cfg, timeout)
    launch_router(cfg, run_dir)
    router_url = f"http://{cfg['router']['host']}:{cfg['router']['port']}"
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline and not healthy(router_url):
        time.sleep(5)
    if not healthy(router_url):
        raise TimeoutError("router did not become healthy")
    print(f"cluster ready: {router_url}/v1")


def stop(cfg: dict, run_dir: Path) -> None:
    targets = [(None, "router")] + [(w["ssh"], w["name"]) for w in cfg["workers"]]
    for host, name in targets:
        pid = run_dir / "pids" / f"{name}.pid"
        run_on(host, f"test ! -f {shlex.quote(str(pid))} || kill $(cat {shlex.quote(str(pid))}) 2>/dev/null || true", check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("deploy", "stop", "wait"))
    parser.add_argument("--config")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action == "deploy":
        deploy(cfg, Path(args.run_dir), args.timeout)
    elif args.action == "wait":
        wait_workers(cfg, args.timeout)
    else:
        stop(cfg, Path(args.run_dir))


if __name__ == "__main__":
    main()
