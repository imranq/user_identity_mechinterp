#!/usr/bin/env python3
"""Summarize A/B logit diffs across prompts for a sweep config.

Reads report_artifacts/cot_diff_curve_*_{config}.csv and writes:
- report_artifacts/ab_summary_{config}.csv
- report_artifacts/ab_baseline_margins_{config}.png (if matplotlib available)

Prints mean/SD/t-stat for delta = steered - baseline.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize A/B diffs for a sweep config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Config tag suffix used in cot_diff_curve_*_{config}.csv",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="report_artifacts",
        help="Directory containing sweep CSVs and where outputs will be written.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip writing baseline margin plot.",
    )
    return parser.parse_args()


def read_last_row(path: str) -> tuple[int, float, float] | None:
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("layer"):
                last = line
    if not last:
        return None
    layer_str, steered_str, baseline_str = last.split(",")
    return int(layer_str), float(steered_str), float(baseline_str)


def main() -> None:
    args = parse_args()
    config = args.config
    artifacts_dir = args.artifacts_dir

    pattern = os.path.join(artifacts_dir, f"cot_diff_curve_*_{config}.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")

    rows: list[tuple[str, float, float, float, bool]] = []
    for path in paths:
        prompt = os.path.basename(path).split("cot_diff_curve_")[1].split(f"_{config}.csv")[0]
        last = read_last_row(path)
        if not last:
            continue
        _, steered, baseline = last
        delta = steered - baseline
        flipped = steered < 0 and baseline > 0
        rows.append((prompt, baseline, steered, delta, flipped))

    rows.sort(key=lambda r: r[0])

    out_csv = os.path.join(artifacts_dir, f"ab_summary_{config}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "baseline_ab", "steered_ab", "delta", "flipped"])
        w.writerows(rows)

    print(f"Wrote: {out_csv}")
    if rows:
        deltas = [r[3] for r in rows]
        if len(deltas) >= 2:
            m = mean(deltas)
            s = stdev(deltas)
            t = m / (s / math.sqrt(len(deltas)))
            print(f"Mean delta: {m:.3f}")
            print(f"SD delta: {s:.3f}")
            print(f"t-stat (df={len(deltas)-1}): {t:.3f}")
            try:
                import scipy.stats as stats

                p = stats.t.sf(abs(t), df=len(deltas) - 1) * 2
                print(f"p-value: {p:.4f}")
            except Exception:
                print("scipy not available; p-value not computed")
        else:
            print("Not enough data for stats")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}")
        return

    prompts = [r[0] for r in rows]
    baselines = [r[1] for r in rows]
    flips = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["tab:red" if f else "tab:blue" for f in flips]
    ax.bar(range(len(prompts)), baselines, color=colors)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, rotation=45, ha="right")
    ax.set_ylabel("Baseline A-B")
    ax.set_title(f"Baseline margins ({config})")
    fig.tight_layout()
    out_plot = os.path.join(artifacts_dir, f"ab_baseline_margins_{config}.png")
    fig.savefig(out_plot)
    print(f"Wrote: {out_plot}")


if __name__ == "__main__":
    main()
