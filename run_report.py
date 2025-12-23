#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from probe_user_representation import run_probe
from persona_prompts import build_prompt_dataset
from compute_persona_direction import extract_activations, mean_direction, probe_direction
from persona_patching_runner import apply_direction
from autorater_stub import SYSTEM_PROMPT

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["layer"], df["accuracy"], label="accuracy")
    ax.plot(df["layer"], df["auc"], label="auc")
    ax.set_xlabel("layer")
    ax.set_ylabel("metric")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_control_plot(results: Dict[str, pd.DataFrame], out_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, df in results.items():
        ax.plot(df["layer"], df["auc"], label=f"{name} auc")
    ax.set_xlabel("layer")
    ax.set_ylabel("auc")
    ax.set_title("Control AUC by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _best_row(df: pd.DataFrame) -> Dict[str, Any]:
    best = df.loc[df["balanced_accuracy"].idxmax()]
    return {
        "layer": int(best["layer"]),
        "accuracy": float(best["accuracy"]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
        "auc": float(best["auc"]),
        "best_threshold": float(best["best_threshold"]),
    }


def _run_probe_variant(
    name: str,
    out_dir: Path,
    model_name: str,
    seed: int,
    n_questions_per_pair: int,
    template_holdout: bool,
    question_holdout: bool,
    max_layers: int,
    device: str,
    min_layer: int,
    probe_position: str,
    align_probe_index: bool,
    probe_template_id: int,
    persona_pad_token: str,
    drop_persona: bool,
    shuffle_labels: bool,
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print(f"\n=== Probe: {name} ===")
    df = run_probe(
        model_name=model_name,
        seed=seed,
        n_questions_per_pair=n_questions_per_pair,
        template_holdout=template_holdout,
        question_holdout=question_holdout,
        max_layers=max_layers,
        device=device,
        min_layer=min_layer,
        show_tokens=False,
        show_count=0,
        show_vector=False,
        show_vector_layer=0,
        show_examples=False,
        show_examples_count=0,
        show_embedding_table=False,
        show_embedding_table_rows=0,
        show_embedding_table_dims=0,
        show_timing=True,
        probe_position=probe_position,
        align_persona_lengths=False,
        pad_token=persona_pad_token,
        align_probe_index=align_probe_index,
        probe_template_id=probe_template_id,
        drop_persona=drop_persona,
        shuffle_labels=shuffle_labels,
        batch_size=batch_size,
        model=None,
    )
    csv_path = out_dir / f"{name}_probe_results.csv"
    df.to_csv(csv_path, index=False)
    _save_plot(df, out_dir / f"{name}_probe_plot.png", f"{name} probe")
    summary = _best_row(df)
    summary["csv"] = str(csv_path)
    return df, summary


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows))


def run_report(
    output_dir: str = "report_artifacts",
    model_name: str = "google/gemma-2-2b-it",
    seed: int = 42,
    n_questions_per_pair: int = 10,
    max_layers: int = 12,
    min_layer: int = 1,
    device: str = "0",
    batch_size: int = 32,
    probe_position: str = "question_last_token",
    align_probe_index: bool = True,
    probe_template_id: int = 0,
    persona_pad_token: str = " X",
    direction_layer: int = 4,
    direction_method: str = "mean",
    alpha: float = 3.0,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    summary: Dict[str, Any] = {
        "config": {
            "model_name": model_name,
            "seed": seed,
            "n_questions_per_pair": n_questions_per_pair,
            "max_layers": max_layers,
            "min_layer": min_layer,
            "device": device,
            "batch_size": batch_size,
            "probe_position": probe_position,
            "align_probe_index": align_probe_index,
            "probe_template_id": probe_template_id,
            "persona_pad_token": persona_pad_token,
            "direction_layer": direction_layer,
            "direction_method": direction_method,
            "alpha": alpha,
        },
        "artifacts": {},
        "results": {},
        "warnings": [],
    }

    # Baseline and controls
    baseline_df, baseline_summary = _run_probe_variant(
        "baseline",
        out_dir,
        model_name,
        seed,
        n_questions_per_pair,
        template_holdout=True,
        question_holdout=False,
        max_layers=max_layers,
        device=device,
        min_layer=min_layer,
        probe_position=probe_position,
        align_probe_index=align_probe_index,
        probe_template_id=probe_template_id,
        persona_pad_token=persona_pad_token,
        drop_persona=False,
        shuffle_labels=False,
        batch_size=batch_size,
    )
    summary["results"]["baseline"] = baseline_summary

    controls: Dict[str, pd.DataFrame] = {}
    drop_df, drop_summary = _run_probe_variant(
        "drop_persona",
        out_dir,
        model_name,
        seed,
        n_questions_per_pair,
        template_holdout=True,
        question_holdout=False,
        max_layers=max_layers,
        device=device,
        min_layer=min_layer,
        probe_position=probe_position,
        align_probe_index=align_probe_index,
        probe_template_id=probe_template_id,
        persona_pad_token=persona_pad_token,
        drop_persona=True,
        shuffle_labels=False,
        batch_size=batch_size,
    )
    controls["drop_persona"] = drop_df
    summary["results"]["drop_persona"] = drop_summary

    shuffle_df, shuffle_summary = _run_probe_variant(
        "shuffle_labels",
        out_dir,
        model_name,
        seed,
        n_questions_per_pair,
        template_holdout=True,
        question_holdout=False,
        max_layers=max_layers,
        device=device,
        min_layer=min_layer,
        probe_position=probe_position,
        align_probe_index=align_probe_index,
        probe_template_id=probe_template_id,
        persona_pad_token=persona_pad_token,
        drop_persona=False,
        shuffle_labels=True,
        batch_size=batch_size,
    )
    controls["shuffle_labels"] = shuffle_df
    summary["results"]["shuffle_labels"] = shuffle_summary

    question_df, question_summary = _run_probe_variant(
        "question_holdout",
        out_dir,
        model_name,
        seed,
        n_questions_per_pair,
        template_holdout=False,
        question_holdout=True,
        max_layers=max_layers,
        device=device,
        min_layer=min_layer,
        probe_position=probe_position,
        align_probe_index=align_probe_index,
        probe_template_id=probe_template_id,
        persona_pad_token=persona_pad_token,
        drop_persona=False,
        shuffle_labels=False,
        batch_size=batch_size,
    )
    controls["question_holdout"] = question_df
    summary["results"]["question_holdout"] = question_summary

    _save_control_plot(controls, out_dir / "controls_auc_plot.png")
    if plt is None:
        summary["warnings"].append("matplotlib not available; plots were skipped")

    # Direction extraction
    print("\n=== Direction extraction ===")
    if device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif str(device).isdigit():
        device_str = f"cuda:{device}"
    else:
        device_str = str(device)
    model = HookedTransformer.from_pretrained(model_name, device=device_str)
    prompt_objs = build_prompt_dataset(
        n_questions_per_pair=n_questions_per_pair,
        seed=seed,
        align_persona_lengths=False,
        tokenizer=model if align_probe_index else None,
        pad_token=persona_pad_token,
        align_probe_index=align_probe_index,
        probe_template_id=probe_template_id,
        drop_persona=False,
    )
    prompts = [(p.prompt, p.label) for p in prompt_objs]
    X, y = extract_activations(model, prompts, direction_layer, probe_position)
    if direction_method == "mean":
        direction = mean_direction(X, y)
    else:
        direction = probe_direction(X, y, seed)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    direction_path = out_dir / "persona_direction.npy"
    meta_path = out_dir / "persona_direction.json"
    np.save(direction_path, direction)
    meta = {
        "model_name": model_name,
        "layer": direction_layer,
        "probe_position": probe_position,
        "method": direction_method,
        "n_examples": len(prompts),
        "align_probe_index": align_probe_index,
        "probe_template_id": probe_template_id,
        "persona_pad_token": persona_pad_token,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    summary["artifacts"]["persona_direction"] = str(direction_path)
    summary["artifacts"]["persona_direction_meta"] = str(meta_path)

    # Patching
    print("\n=== Patching ===")
    outputs: List[Dict[str, Any]] = []
    for item in prompt_objs:
        tokens = model.to_tokens(item.prompt)
        clean_tokens = model.generate(tokens, max_new_tokens=80, do_sample=False)
        clean_text = model.to_string(clean_tokens[0])
        patched_text = apply_direction(
            model,
            item.prompt,
            direction,
            direction_layer,
            probe_position,
            alpha,
        )
        outputs.append(
            {
                "pair_id": item.pair_id,
                "role": item.role,
                "label": item.label,
                "prompt": item.prompt,
                "clean_output": clean_text,
                "patched_output": patched_text,
                "alpha": alpha,
                "layer": direction_layer,
                "probe_position": probe_position,
            }
        )
    patched_path = out_dir / "patched_outputs.jsonl"
    _write_jsonl(patched_path, outputs)
    summary["artifacts"]["patched_outputs"] = str(patched_path)

    # Autorater prep
    print("\n=== Autorater prep ===")
    autorater_path = out_dir / "autorater_inputs.csv"
    rows = []
    for idx, row in enumerate(outputs):
        rows.append(
            {
                "id": idx,
                "system_prompt": SYSTEM_PROMPT,
                "prompt": row["prompt"],
                "clean_output": row["clean_output"],
                "patched_output": row["patched_output"],
            }
        )
    pd.DataFrame(rows).to_csv(autorater_path, index=False)
    summary["artifacts"]["autorater_inputs"] = str(autorater_path)

    summary["artifacts"]["controls_plot"] = str(out_dir / "controls_auc_plot.png")
    summary["artifacts"]["baseline_plot"] = str(out_dir / "baseline_probe_plot.png")
    summary["elapsed_seconds"] = time.perf_counter() - start_time

    summary_path = out_dir / "report_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\nSaved report summary to", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full report pipeline")
    parser.add_argument("--output_dir", type=str, default="report_artifacts")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_questions_per_pair", type=int, default=10)
    parser.add_argument("--max_layers", type=int, default=12)
    parser.add_argument("--min_layer", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--probe_position",
        type=str,
        default="question_last_token",
        choices=["persona", "question_end", "question_last_token", "prompt_end"],
    )
    parser.add_argument("--align_probe_index", action="store_true")
    parser.add_argument("--probe_template_id", type=int, default=0)
    parser.add_argument("--persona_pad_token", type=str, default=" X")
    parser.add_argument("--direction_layer", type=int, default=4)
    parser.add_argument(
        "--direction_method",
        type=str,
        default="mean",
        choices=["mean", "probe"],
    )
    parser.add_argument("--alpha", type=float, default=3.0)
    args = parser.parse_args()

    run_report(
        output_dir=args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        n_questions_per_pair=args.n_questions_per_pair,
        max_layers=args.max_layers,
        min_layer=args.min_layer,
        device=args.device,
        batch_size=args.batch_size,
        probe_position=args.probe_position,
        align_probe_index=args.align_probe_index,
        probe_template_id=args.probe_template_id,
        persona_pad_token=args.persona_pad_token,
        direction_layer=args.direction_layer,
        direction_method=args.direction_method,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
