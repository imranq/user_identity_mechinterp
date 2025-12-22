"""
This script serves as a command-line interface to run the various experiments defined in the other files
of this project. It allows the user to easily execute the persona probing, activation patching, and
Chain-of-Thought (CoT) faithfulness experiments, either individually or all at once.

The script imports the necessary functions from the other modules and wraps them in functions that
handle argument passing and result printing. This provides a centralized and convenient way to
manage and run the entire suite of experiments.
"""

import argparse

from probe_user_representation import run_probe
from activation_patching import patch_persona_activation
from persona_prompts import build_prompts
from cot_faithfulness import logit_lens_scores, PUZZLES
from transformer_lens import HookedTransformer
import numpy as np


def run_probe_experiment(model_name: str, seed: int, save_path: str) -> None:
    """
    Runs the linear probe experiment and saves the results.

    This function is a wrapper around the `run_probe` function from `probe_user_representation.py`.

    Args:
        model_name: The name of the model to use.
        seed: The random seed for reproducibility.
        save_path: The path to save the CSV file with the results.
    """
    df = run_probe(model_name, seed)
    df.to_csv(save_path, index=False)
    best_row = df.loc[df["accuracy"].idxmax()]
    print("\n--- Probe Experiment Results ---")
    print("Probe results saved to", save_path)
    print(f"Best layer for persona representation: {int(best_row['layer'])} (Accuracy: {best_row['accuracy']:.4f})")


def run_patch_experiment(model_name: str, pair_id: str, layer: int) -> None:
    """
    Runs the activation patching experiment.

    This function is a wrapper around the `patch_persona_activation` function from `activation_patching.py`.

    Args:
        model_name: The name of the model to use.
        pair_id: The ID of the persona pair to use.
        layer: The layer to perform the patching on.
    """
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

    print("\n--- Activation Patching Experiment Results ---")
    print("Pair:", pair_id)
    print("Layer:", layer)
    print(f"Clean logit diff ('formal' - 'simple'): {results['clean_score']:.4f}")
    print(f"Patched logit diff ('formal' - 'simple'): {results['patched_score']:.4f}")
    print(f"Change in logit diff due to patching: {results['delta']:.4f}")


def run_cot_experiment(model_name: str, use_hint: bool) -> None:
    """
    Runs the Chain-of-Thought / logit lens experiment.

    This function is a wrapper around the `logit_lens_scores` function from `cot_faithfulness.py`.

    Args:
        model_name: The name of the model to use.
        use_hint: Whether to include the misleading hint in the prompt.
    """
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    for puzzle in PUZZLES:
        prefix = puzzle["hinted"] if use_hint else ""
        prompt = f"{prefix}{puzzle['question']}\nAnswer:"
        scores = logit_lens_scores(model, prompt, puzzle["choices"])
        diffs = np.array(scores[puzzle["choices"][0]]) - np.array(scores[puzzle["choices"][1]])
        pivot_layer = int(np.argmax(np.abs(diffs))) # Pivot is where the decision is sharpest
        
        print("\n--- CoT / Logit Lens Experiment Results ---")
        print("Puzzle:", puzzle["id"])
        print("Hinted:", use_hint)
        print(f"Pivot layer (max absolute difference): {pivot_layer}")
        print(f"Final layer logit diff ('{puzzle['choices'][0']}' - '{puzzle['choices'][1']}'): {diffs[-1]:.4f}")


def main() -> None:
    """
    Main function that parses command-line arguments and runs the selected experiments.
    """
    parser = argparse.ArgumentParser(description="Run user identity mechanism interpretability experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["probe", "patch", "cot", "all"],
                        help="Which experiment to run: 'probe', 'patch', 'cot', or 'all'.")
    
    # Arguments for all experiments
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model to use (e.g., 'gpt2', 'gpt2-medium').")
    
    # Arguments for the probe experiment
    parser.add_argument("--seed", type=int, default=42, help="Random seed for probe experiment.")
    parser.add_argument("--save_path", type=str, default="probe_results.csv", help="Path to save probe results.")

    # Arguments for the activation patching experiment
    parser.add_argument("--pair_id", type=str, default="physics", help="Persona pair ID for patching (e.g., 'physics').")
    parser.add_argument("--layer", type=int, default=5, help="Layer to use for activation patching.")

    # Arguments for the CoT experiment
    parser.add_argument("--use_hint", action="store_true", help="Use misleading hint in the CoT puzzle.")
    
    args = parser.parse_args()

    print(f"Running experiment(s): '{args.experiment}' with model '{args.model_name}'")

    if args.experiment in ["probe", "all"]:
        run_probe_experiment(args.model_name, args.seed, args.save_path)

    if args.experiment in ["patch", "all"]:
        # As a default, if running all experiments, let's use the best layer from the probe.
        layer_for_patching = args.layer
        if args.experiment == "all":
            try:
                import pandas as pd
                probe_df = pd.read_csv(args.save_path)
                layer_for_patching = int(probe_df.loc[probe_df["accuracy"].idxmax()]["layer"])
                print(f"\nUsing best layer from probe for patching: Layer {layer_for_patching}")
            except FileNotFoundError:
                print(f"Could not find probe results at {args.save_path}, using default layer {args.layer} for patching.")
        
        run_patch_experiment(args.model_name, args.pair_id, layer_for_patching)

    if args.experiment in ["cot", "all"]:
        run_cot_experiment(args.model_name, args.use_hint)


if __name__ == "__main__":
    main()
