#!/usr/bin/env python3
"""Generate token-pool usage charts and a Markdown report for a 2P1D run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEVELS = (16, 32, 48, 64)
NODES = ("p0", "p1", "d0")
COLORS = {"p0": "#0072B2", "p1": "#009E73", "d0": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric_values(sample: dict[str, Any], node: str, prefix: str) -> list[float]:
    return [
        float(item["value"])
        for item in sample.get("sglang", {}).get(node, [])
        if item.get("metric", "").startswith(prefix)
    ]


def mean_metric(sample: dict[str, Any], node: str, prefix: str) -> float:
    values = metric_values(sample, node, prefix)
    return float(np.mean(values)) if values else np.nan


def scalar_metric(sample: dict[str, Any], node: str, prefix: str) -> float:
    values = metric_values(sample, node, prefix)
    return values[0] if values else np.nan


def load_samples(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            row: dict[str, Any] = {
                "timestamp": pd.Timestamp(sample["timestamp"]),
                "sequence": int(sample["sequence"]),
                "monitor_errors": len(sample.get("errors", [])),
            }
            for node in NODES:
                row[f"{node}_usage"] = mean_metric(
                    sample, node, "sglang:full_token_usage{"
                )
                row[f"{node}_used_tokens"] = mean_metric(
                    sample, node, "sglang:num_used_tokens{"
                )
                row[f"{node}_pending_usage"] = mean_metric(
                    sample, node, "sglang:pending_prealloc_token_usage{"
                )
                row[f"{node}_prompt_total"] = scalar_metric(
                    sample, node, "sglang:prompt_tokens_total{"
                )
                row[f"{node}_generation_total"] = scalar_metric(
                    sample, node, "sglang:generation_tokens_total{"
                )
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    frame["prefill_mean_usage"] = frame[["p0_usage", "p1_usage"]].mean(axis=1)
    return frame


def load_windows(run_dir: Path) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    windows: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for level in LEVELS:
        timing = load_json(run_dir / f"concurrency-{level}" / "timing.json")
        windows[level] = (
            pd.Timestamp(timing["started_at"]),
            pd.Timestamp(timing["ended_at"]),
        )
    return windows


def assign_levels(
    frame: pd.DataFrame, windows: dict[int, tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.DataFrame:
    frame = frame.copy()
    frame["concurrency"] = pd.NA
    for level, (started, ended) in windows.items():
        # The final monitor flush can land a few seconds after timing.json.
        mask = (frame["timestamp"] >= started) & (
            frame["timestamp"] <= ended + pd.Timedelta(seconds=60)
        )
        frame.loc[mask, "concurrency"] = level
    frame["concurrency"] = frame["concurrency"].astype("Int64")
    return frame


def level_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        part = frame[frame["concurrency"] == level]
        active = part[["p0_usage", "p1_usage", "d0_usage"]].max(axis=1) > 1e-4
        prefill_peak_index = part["prefill_mean_usage"].idxmax()
        decode_peak_index = part["d0_usage"].idxmax()
        rows.append(
            {
                "concurrency": level,
                "samples": len(part),
                "active_samples": int(active.sum()),
                "prefill_mean": part["prefill_mean_usage"].mean(),
                "prefill_peak": part.loc[prefill_peak_index, "prefill_mean_usage"],
                "prefill_peak_time": part.loc[prefill_peak_index, "timestamp"],
                "decode_mean": part["d0_usage"].mean(),
                "decode_peak": part.loc[decode_peak_index, "d0_usage"],
                "decode_peak_time": part.loc[decode_peak_index, "timestamp"],
                "decode_peak_used_tokens": part.loc[
                    decode_peak_index, "d0_used_tokens"
                ],
            }
        )
    return pd.DataFrame(rows).set_index("concurrency")


def node_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        part = frame[frame["concurrency"] == level]
        for node in NODES:
            series = part[f"{node}_usage"]
            nonzero = series > 1e-4
            peak_index = series.idxmax()
            rows.append(
                {
                    "concurrency": level,
                    "node": node,
                    "samples": len(series),
                    "nonzero_samples": int(nonzero.sum()),
                    "mean": series.mean(),
                    "active_mean": series[nonzero].mean() if nonzero.any() else 0.0,
                    "peak": series.loc[peak_index],
                    "peak_time": part.loc[peak_index, "timestamp"],
                    "peak_used_tokens": part.loc[peak_index, f"{node}_used_tokens"],
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_timeline(
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    windows: dict[int, tuple[pd.Timestamp, pd.Timestamp]],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.4))
    shades = ("#EAF2F8", "#F2F8EA", "#FFF4E5", "#F4ECF7")
    for shade, level in zip(shades, LEVELS):
        started, ended = windows[level]
        ax.axvspan(started, ended, color=shade, alpha=0.65, zorder=0)
        midpoint = started + (ended - started) / 2
        ax.text(
            midpoint,
            0.985,
            f"Concurrency {level}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#444444",
            fontsize=9,
        )

    for node in NODES:
        ax.plot(
            frame["timestamp"],
            frame[f"{node}_usage"] * 100,
            label={"p0": "Prefill P0", "p1": "Prefill P1", "d0": "Decode D0"}[node],
            color=COLORS[node],
            linewidth=2.2 if node == "d0" else 1.8,
            marker="o",
            markersize=3.5,
        )

    for level, row in stats.iterrows():
        offset = -18 if row["decode_peak"] > 0.93 else 9
        ax.annotate(
            f"{row['decode_peak'] * 100:.1f}%",
            (row["decode_peak_time"], row["decode_peak"] * 100),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="top" if offset < 0 else "bottom",
            fontsize=9,
            color=COLORS["d0"],
            fontweight="bold",
        )

    ax.set_title("GLM-5.2 2P1D Token Block Usage Over Time", pad=55, fontsize=15)
    ax.set_ylabel("full_token_usage (%)")
    ax.set_xlabel("UTC time on 2026-07-14 (5-minute snapshots)")
    ax.set_ylim(-2, 106)
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=frame["timestamp"].dt.tz))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncols=3,
        frameon=True,
    )
    fig.autofmt_xdate(rotation=0)
    save_figure(fig, output_dir, "token_usage_timeline")


def plot_level_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    means = []
    peaks = []
    for level in LEVELS:
        part = frame[frame["concurrency"] == level]
        means.append([part[f"{node}_usage"].mean() * 100 for node in NODES])
        peaks.append([part[f"{node}_usage"].max() * 100 for node in NODES])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    x = np.arange(len(LEVELS))
    width = 0.24
    for ax, values, title in zip(
        axes,
        (np.asarray(means), np.asarray(peaks)),
        ("Window mean (includes idle repair periods)", "Peak snapshot"),
    ):
        for index, node in enumerate(NODES):
            bars = ax.bar(
                x + (index - 1) * width,
                values[:, index],
                width,
                label=node.upper(),
                color=COLORS[node],
            )
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x, [str(level) for level in LEVELS])
        ax.set_xlabel("Concurrency")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("full_token_usage (%)")
    axes[1].legend(loc="upper left")
    axes[0].set_ylim(0, 106)
    fig.suptitle("Token Block Usage by Concurrency", fontsize=15)
    save_figure(fig, output_dir, "token_usage_by_concurrency")


def plot_counters(frame: pd.DataFrame, output_dir: Path) -> None:
    baseline = frame.iloc[0]
    fig, ax = plt.subplots(figsize=(14, 5.8))
    for node in NODES:
        delta = (frame[f"{node}_prompt_total"] - baseline[f"{node}_prompt_total"]) / 1e6
        ax.plot(
            frame["timestamp"],
            delta,
            label=f"{node.upper()} prompt tokens",
            color=COLORS[node],
            linewidth=2,
        )
    ax2 = ax.twinx()
    generation_delta = (
        frame["d0_generation_total"] - baseline["d0_generation_total"]
    ) / 1e6
    ax2.plot(
        frame["timestamp"],
        generation_delta,
        label="D0 generation tokens",
        color="#CC79A7",
        linestyle="--",
        linewidth=2,
    )
    ax.set_title("Cumulative Token Counters During the Full Suite", fontsize=15)
    ax.set_xlabel("UTC time on 2026-07-14")
    ax.set_ylabel("Prompt token delta (million)")
    ax2.set_ylabel("D0 generation token delta (million)")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=frame["timestamp"].dt.tz))
    ax.grid(axis="y", alpha=0.25)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left")
    save_figure(fig, output_dir, "token_counters_timeline")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_time(value: pd.Timestamp) -> str:
    return value.strftime("%H:%M:%S")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("|" + "|".join("---" for _ in headers) + "|")
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(output)


def write_report(
    run_dir: Path,
    output_dir: Path,
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    detailed: pd.DataFrame,
    windows: dict[int, tuple[pd.Timestamp, pd.Timestamp]],
) -> None:
    suite = load_json(run_dir / "suite_summary.json")
    level_rows: list[list[Any]] = []
    performance_rows: list[list[Any]] = []
    for level in LEVELS:
        row = stats.loc[level]
        summary = suite[f"concurrency-{level}"]
        started, ended = windows[level]
        level_rows.append(
            [
                level,
                f"{started:%H:%M}–{ended:%H:%M}",
                f"{row['active_samples']}/{row['samples']}",
                pct(row["prefill_mean"]),
                pct(row["prefill_peak"]),
                pct(row["decode_mean"]),
                pct(row["decode_peak"]),
                f"{int(row['decode_peak_used_tokens']):,}",
            ]
        )
        metrics = summary["request_metrics"]
        performance_rows.append(
            [
                level,
                summary["tasks_with_trajectory"],
                f"{summary['successful_requests']:,}",
                f"{summary['request_errors']:,}",
                f"{metrics['ttft_ms']['avg'] / 1000:.2f} s",
                f"{metrics['tpot_ms']['avg']:.2f} ms",
                f"{summary['aggregate_cache_hit_rate'] * 100:.2f}%",
            ]
        )

    detail_rows: list[list[Any]] = []
    for row in detailed.itertuples(index=False):
        detail_rows.append(
            [
                row.concurrency,
                row.node.upper(),
                f"{row.nonzero_samples}/{row.samples}",
                pct(row.mean),
                pct(row.active_mean),
                pct(row.peak),
                fmt_time(row.peak_time),
                f"{int(row.peak_used_tokens):,}",
            ]
        )

    first = frame.iloc[0]
    last = frame.iloc[-1]
    counter_rows = []
    for node in NODES:
        counter_rows.append(
            [
                node.upper(),
                f"{last[f'{node}_prompt_total'] - first[f'{node}_prompt_total']:,.0f}",
                f"{last[f'{node}_generation_total'] - first[f'{node}_generation_total']:,.0f}",
            ]
        )

    report = f"""# GLM-5.2 2P1D Token Usage 时序分析

运行 ID：`{run_dir.name}`  
采样范围：`{frame.iloc[0]['timestamp'].isoformat()}` 至 `{frame.iloc[-1]['timestamp'].isoformat()}`  
采样间隔：5 分钟，共 {len(frame)} 个采样点；监控采样错误数：{int(frame['monitor_errors'].sum())}。

## 核心结论

- `full_token_usage` 表示 SGLang token/KV block 池的瞬时占用率，不是 token/s 吞吐。
- Decode D0 是主要容量压力点：并发 32、48、64 的峰值分别达到 {pct(stats.loc[32, 'decode_peak'])}、{pct(stats.loc[48, 'decode_peak'])}、{pct(stats.loc[64, 'decode_peak'])}。
- Prefill P0/P1 呈短促脉冲，长期累计 prompt token 分配接近 50/50；Prefill 完成后会传输并释放 KV，而 Decode 需要保留长对话 KV。
- 全部节点所有 `pending_prealloc_token_usage` 采样均为 0，没有发现持续的 token block 预分配积压。
- 每档后半段的大量零值主要来自 41 个任务在预处理阶段持续补跑失败，不代表 2P1D 服务中断。
- 5 分钟采样可能漏掉短于采样周期的 Prefill 峰值，因此 Prefill 均值应视为稀疏快照统计。

## 完整时序

![Token usage timeline](token_usage_timeline.png)

并发档位背景色依据每档 `timing.json` 标注；Decode 峰值已直接标在曲线上。

## 分档对比

![Token usage by concurrency](token_usage_by_concurrency.png)

{markdown_table(
    ["并发", "UTC 时间", "活跃采样", "Prefill 均值", "Prefill 峰值", "Decode 均值", "Decode 峰值", "Decode 峰值 tokens"],
    level_rows,
)}

窗口均值包含补跑不可运行任务时的空闲阶段，因此显著低于活跃阶段。

## 节点明细

{markdown_table(
    ["并发", "节点", "非零采样", "窗口均值", "非零均值", "峰值", "峰值时间 UTC", "峰值 used tokens"],
    detail_rows,
)}

## 累计 Token Counter

![Cumulative token counters](token_counters_timeline.png)

{markdown_table(["节点", "Prompt token 增量", "Generation token 增量"], counter_rows)}

P0 与 P1 的 prompt token 增量差异很小，D0 的 prompt token 增量接近两者之和；D0 承担了几乎全部 generation token。

## 与请求性能的对应关系

{markdown_table(
    ["并发", "完整 trajectory", "成功请求", "请求错误", "平均 TTFT", "平均 TPOT", "Cache hit"],
    performance_rows,
)}

随着 Decode token pool 从并发 32 开始频繁达到 95%–99%，平均 TTFT 从并发 16 的 2.19 秒上升到并发 64 的 11.46 秒。两者趋势一致，但由于 5 分钟采样较稀疏，不能仅凭该图证明严格因果关系。

## 数据与方法

- 原始采样：`{run_dir / 'monitor' / 'samples.jsonl'}`
- 请求与聚合指标：`{run_dir / 'suite_summary.json'}`
- 分档时间：各 `concurrency-*/timing.json`
- 本目录处理后时序：[`token_usage_timeseries.csv`](token_usage_timeseries.csv)
- SVG 矢量图：[`timeline`](token_usage_timeline.svg)、[`concurrency comparison`](token_usage_by_concurrency.svg)、[`counters`](token_counters_timeline.svg)
- 生成脚本：`experiments/glm52_2p1d/plot_token_usage.py`
"""
    (output_dir / "token_usage_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_samples(run_dir / "monitor" / "samples.jsonl")
    windows = load_windows(run_dir)
    frame = assign_levels(frame, windows)
    stats = level_stats(frame)
    detailed = node_stats(frame)

    frame.to_csv(output_dir / "token_usage_timeseries.csv", index=False)
    plot_timeline(frame, stats, windows, output_dir)
    plot_level_summary(frame, output_dir)
    plot_counters(frame, output_dir)
    write_report(run_dir, output_dir, frame, stats, detailed, windows)

    print(output_dir / "token_usage_report.md")


if __name__ == "__main__":
    main()
