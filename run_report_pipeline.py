#!/usr/bin/env python3
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "report_artifacts"

MODEL_NAME = "google/gemma-2-2b-it"
PROBE_POSITION = "question_last_token"
N_QUESTIONS = 10
MAX_LAYERS = 12
MIN_LAYER = 1
BATCH_SIZE = 32

STEER_LAYERS = "16,18,20"
STEER_ALPHAS = "8.0,12.0,16.0"
STEER_TEMPERATURE = "0.7"
STEER_TOP_P = "0.9"


def run(cmd: List[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_layer(rows: List[Dict[str, str]]) -> Dict[str, str]:
    best = max(rows, key=lambda r: float(r["balanced_accuracy"]))
    return best


def plot_probe(csv_path: Path, out_path: Path, title: str) -> None:
    rows = read_csv(csv_path)
    layers = [int(r["layer"]) for r in rows]
    acc = [float(r["accuracy"]) for r in rows]
    auc = [float(r["auc"]) for r in rows]
    plt.figure(figsize=(7, 3))
    plt.plot(layers, acc, label="accuracy")
    plt.plot(layers, auc, label="auc")
    plt.xlabel("layer")
    plt.ylabel("metric")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_controls(control_paths: Dict[str, Path], out_path: Path) -> None:
    plt.figure(figsize=(7, 3))
    for name, path in control_paths.items():
        rows = read_csv(path)
        layers = [int(r["layer"]) for r in rows]
        auc = [float(r["auc"]) for r in rows]
        plt.plot(layers, auc, label=f"{name} auc")
    plt.xlabel("layer")
    plt.ylabel("auc")
    plt.title("Control AUC by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_report(summary: Dict[str, str], out_path: Path) -> None:
    out_path.write_text(
        "\n".join(
            [
                "# Report (Auto-generated)",
                "",
                "## Summary",
                f"- Baseline best layer: {summary['baseline_best']}",
                f"- Drop-persona best layer: {summary['drop_best']}",
                f"- Shuffle-labels best layer: {summary['shuffle_best']}",
                f"- Question-holdout best layer: {summary['question_best']}",
                "",
                "## Artifacts",
                "- baseline_probe.csv",
                "- drop_persona_probe.csv",
                "- shuffle_labels_probe.csv",
                "- question_holdout_probe.csv",
                "- baseline_probe_plot.png",
                "- controls_auc_plot.png",
                "- persona_direction.npy / persona_direction.json",
                "- persona_steer_outputs.jsonl",
                "- persona_steer_outputs_random.jsonl",
                "- steer_kl.csv / steer_kl_random.csv",
                "- cot_lens_switches_hinted_False.png",
                "- cot_lens_switches_hinted_True.png",
                "- cot_lens_switches_hinted_False_steered.png",
                "- cot_lens_switches_hinted_True_steered.png",
                "- cot_kl_switches_hinted_False.png",
                "- cot_kl_switches_hinted_True.png",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # 1) Baseline probe
    run(
        [
            "python",
            "run_all.py",
            "--experiment",
            "probe",
            "--model_name",
            MODEL_NAME,
            "--n_questions_per_pair",
            str(N_QUESTIONS),
            "--template_holdout",
            "--max_layers",
            str(MAX_LAYERS),
            "--min_layer",
            str(MIN_LAYER),
            "--device",
            "0",
            "--probe_position",
            PROBE_POSITION,
            "--align_probe_index",
            "--batch_size",
            str(BATCH_SIZE),
            "--save_path",
            str(ARTIFACTS / "baseline_probe.csv"),
        ]
    )
    plot_probe(ARTIFACTS / "baseline_probe.csv", ARTIFACTS / "baseline_probe_plot.png", "Baseline probe")

    # 2) Controls
    run(
        [
            "python",
            "run_all.py",
            "--experiment",
            "probe",
            "--model_name",
            MODEL_NAME,
            "--n_questions_per_pair",
            str(N_QUESTIONS),
            "--template_holdout",
            "--max_layers",
            str(MAX_LAYERS),
            "--min_layer",
            str(MIN_LAYER),
            "--device",
            "0",
            "--probe_position",
            PROBE_POSITION,
            "--align_probe_index",
            "--batch_size",
            str(BATCH_SIZE),
            "--drop_persona",
            "--save_path",
            str(ARTIFACTS / "drop_persona_probe.csv"),
        ]
    )
    run(
        [
            "python",
            "run_all.py",
            "--experiment",
            "probe",
            "--model_name",
            MODEL_NAME,
            "--n_questions_per_pair",
            str(N_QUESTIONS),
            "--template_holdout",
            "--max_layers",
            str(MAX_LAYERS),
            "--min_layer",
            str(MIN_LAYER),
            "--device",
            "0",
            "--probe_position",
            PROBE_POSITION,
            "--align_probe_index",
            "--batch_size",
            str(BATCH_SIZE),
            "--shuffle_labels",
            "--save_path",
            str(ARTIFACTS / "shuffle_labels_probe.csv"),
        ]
    )
    run(
        [
            "python",
            "run_all.py",
            "--experiment",
            "probe",
            "--model_name",
            MODEL_NAME,
            "--n_questions_per_pair",
            str(N_QUESTIONS),
            "--max_layers",
            str(MAX_LAYERS),
            "--min_layer",
            str(MIN_LAYER),
            "--device",
            "0",
            "--probe_position",
            PROBE_POSITION,
            "--align_probe_index",
            "--batch_size",
            str(BATCH_SIZE),
            "--question_holdout",
            "--save_path",
            str(ARTIFACTS / "question_holdout_probe.csv"),
        ]
    )
    plot_controls(
        {
            "drop_persona": ARTIFACTS / "drop_persona_probe.csv",
            "shuffle_labels": ARTIFACTS / "shuffle_labels_probe.csv",
            "question_holdout": ARTIFACTS / "question_holdout_probe.csv",
        },
        ARTIFACTS / "controls_auc_plot.png",
    )

    # 3) Persona direction
    run(
        [
            "python",
            "compute_persona_direction.py",
            "--layer",
            "4",
            "--probe_position",
            PROBE_POSITION,
            "--align_probe_index",
            "--method",
            "mean",
            "--save_path",
            str(ARTIFACTS / "persona_direction.npy"),
            "--meta_path",
            str(ARTIFACTS / "persona_direction.json"),
        ]
    )

    # 4) Steering with sampling, drop persona, KL
    run(
        [
            "python",
            "persona_steer_demo.py",
            "--direction_path",
            str(ARTIFACTS / "persona_direction.npy"),
            "--layers",
            STEER_LAYERS,
            "--alphas",
            STEER_ALPHAS,
            "--max_new_tokens",
            "120",
            "--plot_dir",
            str(ARTIFACTS),
            "--drop_persona",
            "--do_sample",
            "--temperature",
            STEER_TEMPERATURE,
            "--top_p",
            STEER_TOP_P,
            "--prompts_path",
            str(ROOT / "steer_prompts.txt"),
            "--out_path",
            str(ARTIFACTS / "persona_steer_outputs.jsonl"),
            "--kl_report",
            "--kl_out",
            str(ARTIFACTS / "steer_kl.csv"),
        ]
    )
    run(
        [
            "python",
            "persona_steer_demo.py",
            "--direction_path",
            str(ARTIFACTS / "persona_direction.npy"),
            "--layers",
            STEER_LAYERS,
            "--alphas",
            STEER_ALPHAS,
            "--max_new_tokens",
            "120",
            "--plot_dir",
            str(ARTIFACTS),
            "--drop_persona",
            "--do_sample",
            "--temperature",
            STEER_TEMPERATURE,
            "--top_p",
            STEER_TOP_P,
            "--prompts_path",
            str(ROOT / "steer_prompts.txt"),
            "--out_path",
            str(ARTIFACTS / "persona_steer_outputs_random.jsonl"),
            "--random_direction",
            "--random_seed",
            "0",
            "--kl_report",
            "--kl_out",
            str(ARTIFACTS / "steer_kl_random.csv"),
        ]
    )

    # 5) CoT faithfulness baseline, hinted, and steered overlay + KL
    run(
        [
            "python",
            "cot_faithfulness.py",
            "--model_name",
            MODEL_NAME,
            "--device",
            "cuda",
            "--batch_size",
            "8",
        ]
    )
    run(
        [
            "python",
            "cot_faithfulness.py",
            "--model_name",
            MODEL_NAME,
            "--device",
            "cuda",
            "--batch_size",
            "8",
            "--use_hint",
        ]
    )
    run(
        [
            "python",
            "cot_faithfulness.py",
            "--model_name",
            MODEL_NAME,
            "--device",
            "cuda",
            "--batch_size",
            "8",
            "--steer_direction_path",
            str(ARTIFACTS / "persona_direction.npy"),
            "--steer_layer",
            "4",
            "--steer_alpha",
            "8.0",
            "--kl_plot",
        ]
    )

    # Summaries
    baseline = read_csv(ARTIFACTS / "baseline_probe.csv")
    drop = read_csv(ARTIFACTS / "drop_persona_probe.csv")
    shuffle = read_csv(ARTIFACTS / "shuffle_labels_probe.csv")
    question = read_csv(ARTIFACTS / "question_holdout_probe.csv")

    summary = {
        "baseline_best": json.dumps(best_layer(baseline)),
        "drop_best": json.dumps(best_layer(drop)),
        "shuffle_best": json.dumps(best_layer(shuffle)),
        "question_best": json.dumps(best_layer(question)),
    }
    write_report(summary, ARTIFACTS / "report.md")


if __name__ == "__main__":
    main()
