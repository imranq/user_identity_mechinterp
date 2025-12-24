#!/usr/bin/env python3
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "report_artifacts"

MODEL_NAME = "google/gemma-2-2b-it"
PROBE_POSITION = "question_last_token"

STEER_LAYERS = "16,18,20"
STEER_ALPHAS = "8.0,12.0,16.0"
STEER_TEMPERATURE = "0.7"
STEER_TOP_P = "0.9"


def run(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # 1) Persona direction (mean-diff)
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

    # 2) Steering (persona direction)
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

    # 3) Steering (random direction baseline)
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


if __name__ == "__main__":
    main()
