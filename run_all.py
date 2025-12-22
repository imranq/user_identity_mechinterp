import argparse

from probe_user_representation import run_probe
from activation_patching import patch_persona_activation
from persona_prompts import build_prompts
from cot_faithfulness import logit_lens_scores, PUZZLES
from transformer_lens import HookedTransformer
import numpy as np


def run_probe_experiment(model_name: str, seed: int, save_path: str) -> None:
    df = run_probe(model_name, seed)
    df.to_csv(save_path, index=False)
    best_row = df.loc[df["accuracy"].idxmax()]
    print("Probe results saved to", save_path)
    print("Best layer:", int(best_row["layer"]), "accuracy:", float(best_row["accuracy"]))


def run_patch_experiment(model_name: str, pair_id: str, layer: int) -> None:
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    prompts = [p for p in build_prompts() if p.pair_id == pair_id]
    if len(prompts) != 2:
        raise ValueError("pair_id not found or not exactly two prompts")

    expert_prompt = next(p.prompt for p in prompts if p.role == "expert")
    novice_prompt = next(p.prompt for p in prompts if p.role == "novice")

    results = patch_persona_activation(
        model=model,
        source_prompt=expert_prompt,
        target_prompt=novice_prompt,
        layer=layer,
    )

    print("Activation patching results")
    print("Pair:", pair_id)
    print("Layer:", layer)
    print("Clean logit diff:", results["clean_score"])
    print("Patched logit diff:", results["patched_score"])
    print("Delta:", results["delta"])


def run_cot_experiment(model_name: str, use_hint: bool) -> None:
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    for puzzle in PUZZLES:
        prefix = puzzle["hinted"] if use_hint else ""
        prompt = f"{prefix}{puzzle['question']}\nAnswer:"
        scores = logit_lens_scores(model, prompt, puzzle["choices"])
        diffs = np.array(scores[puzzle["choices"][0]]) - np.array(scores[puzzle["choices"][1]])
        pivot_layer = int(np.argmax(diffs))
        print("Puzzle:", puzzle["id"])
        print("Hinted:", use_hint)
        print("Pivot layer:", pivot_layer)
        print("Final layer diff:", diffs[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run user identity experiments")
    parser.add_argument("--experiment", type=str, default="probe",
                        choices=["probe", "patch", "cot", "all"],
                        help="Which experiment to run")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_id", type=str, default="physics")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="probe_results.csv")
    parser.add_argument("--use_hint", action="store_true")
    args = parser.parse_args()

    if args.experiment in ["probe", "all"]:
        run_probe_experiment(args.model_name, args.seed, args.save_path)

    if args.experiment in ["patch", "all"]:
        run_patch_experiment(args.model_name, args.pair_id, args.layer)

    if args.experiment in ["cot", "all"]:
        run_cot_experiment(args.model_name, args.use_hint)


if __name__ == "__main__":
    main()
