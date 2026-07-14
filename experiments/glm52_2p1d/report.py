#!/usr/bin/env python3
"""Aggregate request, tool-call, GPU and SGLang distributions without deleting raw data."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def quantiles(values: list[float]) -> dict:
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return {"count": 0, "avg": None, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    def pick(q: float) -> float:
        return clean[round((len(clean) - 1) * q)]
    return {"count": len(clean), "avg": statistics.fmean(clean), "min": clean[0], "p50": pick(.5), "p90": pick(.9), "p95": pick(.95), "p99": pick(.99), "max": clean[-1]}


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_tool_names(value, result: Counter) -> None:
    if isinstance(value, dict):
        # Chat Completions stores the callable name one level below each
        # assistant tool-call object. Count it here, then skip that list in
        # the generic recursion so one call cannot be counted twice.
        tool_calls = value.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                name = function.get("name") if isinstance(function, dict) else call.get("name")
                if name:
                    result[str(name)] += 1
        kind = str(value.get("type", ""))
        name = value.get("name")
        if name and kind in {"function_call", "tool_call", "function"}:
            result[str(name)] += 1
        for key, child in value.items():
            if key == "tool_calls":
                continue
            extract_tool_names(child, result)
    elif isinstance(value, list):
        for child in value:
            extract_tool_names(child, result)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_report(level_dir: Path) -> dict:
    attempts = []
    for path in level_dir.rglob("request_metrics.jsonl"):
        for row in read_jsonl(path):
            row["source"] = str(path.relative_to(level_dir))
            attempts.append(row)
    ok = [row for row in attempts if row.get("status") == "ok"]
    requests = {}
    for row in ok:
        requests[row.get("request_group_id") or row["request_id"]] = row
    successful = list(requests.values())

    task_tool_counts = []
    tool_distribution = Counter()
    statuses = Counter()
    for path in level_dir.rglob("traj_log.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        statuses[str(data.get("status", "unknown"))] += 1
        task_tool_counts.append(float(data.get("key_stats", {}).get("tool_calls", 0) or 0))
        extract_tool_names(data.get("messages", []), tool_distribution)

    gpu_rows = []
    sglang_values = defaultdict(list)
    monitor_path = level_dir.parent / "monitor" / "samples.jsonl"
    try:
        timing = json.loads((level_dir / "timing.json").read_text())
    except Exception:
        timing = {}
    for sample in read_jsonl(monitor_path):
        if timing.get("started_at") and sample["timestamp"] < timing["started_at"]:
            continue
        if timing.get("ended_at") and sample["timestamp"] > timing["ended_at"]:
            continue
        for node, gpus in sample.get("gpus", {}).items():
            for gpu in gpus:
                row = {"timestamp": sample["timestamp"], "node": node, **gpu}
                gpu_rows.append(row)
        for node, metrics in sample.get("sglang", {}).items():
            for metric in metrics:
                sglang_values[f"{node}:{metric['metric']}"] .append(metric["value"])

    fields = ("ttft_ms", "tpot_ms", "e2e_ms", "prompt_tokens", "cached_prompt_tokens", "uncached_prompt_tokens", "decode_tokens", "total_tokens")
    summary = {
        "request_attempts": len(attempts),
        "request_errors": sum(row.get("status") != "ok" for row in attempts),
        "successful_requests": len(successful),
        "aggregate_cache_hit_rate": (
            sum(float(row.get("cached_prompt_tokens", 0) or 0) for row in successful)
            / sum(float(row.get("prompt_tokens", 0) or 0) for row in successful)
            if sum(float(row.get("prompt_tokens", 0) or 0) for row in successful) else None
        ),
        "request_metrics": {field: quantiles([row.get(field) for row in successful if row.get(field) is not None]) for field in fields},
        "tasks_with_trajectory": len(task_tool_counts),
        "task_status_distribution": dict(statuses),
        "tool_calls_per_task": quantiles(task_tool_counts),
        "tool_call_distribution": dict(tool_distribution.most_common()),
        "gpu_utilization_distribution": {},
        "gpu_memory_used_mib_distribution": {},
        "sglang_metric_distributions": {name: quantiles(values) for name, values in sorted(sglang_values.items())},
    }
    by_gpu_util, by_gpu_mem = defaultdict(list), defaultdict(list)
    for row in gpu_rows:
        key = f"{row['node']}:gpu{row['index']}"
        try:
            by_gpu_util[key].append(float(row["utilization.gpu"]))
            by_gpu_mem[key].append(float(row["memory.used"]))
        except (KeyError, ValueError):
            pass
    summary["gpu_utilization_distribution"] = {key: quantiles(values) for key, values in sorted(by_gpu_util.items())}
    summary["gpu_memory_used_mib_distribution"] = {key: quantiles(values) for key, values in sorted(by_gpu_mem.items())}

    write_csv(level_dir / "request_metrics.csv", attempts)
    write_csv(level_dir / "gpu_samples.csv", gpu_rows)
    write_csv(level_dir / "tool_call_distribution.csv", [{"tool": k, "count": v} for k, v in tool_distribution.most_common()])
    (level_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    suite = {}
    for level in sorted(run_dir.glob("concurrency-*")):
        suite[level.name] = build_report(level)
    (run_dir / "suite_summary.json").write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: {"successful_requests": value["successful_requests"], "tasks": value["tasks_with_trajectory"]} for key, value in suite.items()}, indent=2))


if __name__ == "__main__":
    main()
