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
import torch


def run_probe_experiment(
    model_name: str,
    seed: int,
    save_path: str,
    n_questions_per_pair: int,
    template_holdout: bool,
    max_layers: int,
    device: str,
    min_layer: int,
    show_probe_tokens: bool,
    show_probe_count: int,
    show_probe_vector: bool,
    show_probe_vector_layer: int,
    show_probe_examples: bool,
    show_probe_examples_count: int,
    show_embedding_table: bool,
    show_embedding_table_rows: int,
    show_embedding_table_dims: int,
    show_timing: bool,
    probe_position: str,
    align_persona_lengths: bool,
    persona_pad_token: str,
    align_probe_index: bool,
    probe_template_id: int,
    drop_persona: bool,
    shuffle_labels: bool,
    question_holdout: bool,
    model: HookedTransformer | None = None,
) -> None:
    """
    Runs the linear probe experiment and saves the results.

    This function is a wrapper around the `run_probe` function from `probe_user_representation.py`.

    Args:
        model_name: The name of the model to use.
        seed: The random seed for reproducibility.
        save_path: The path to save the CSV file with the results.
    """
    df = run_probe(
        model_name,
        seed,
        n_questions_per_pair,
        template_holdout,
        max_layers,
        device,
        min_layer,
        show_probe_tokens,
        show_probe_count,
        show_probe_vector,
        show_probe_vector_layer,
        show_probe_examples,
        show_probe_examples_count,
        show_embedding_table,
        show_embedding_table_rows,
        show_embedding_table_dims,
        show_timing,
        probe_position,
        align_persona_lengths,
        persona_pad_token,
        align_probe_index,
        probe_template_id,
        drop_persona,
        shuffle_labels,
        question_holdout,
        model=model,
    )
    df.to_csv(save_path, index=False)
    best_row = df.loc[df["balanced_accuracy"].idxmax()]
    print("\n--- Probe Experiment Results ---")
    print("Probe results saved to", save_path)
    print(df.to_string(index=False))
    print(
        "Best layer for persona representation:",
        int(best_row["layer"]),
        f"(Balanced Acc: {best_row['balanced_accuracy']:.4f}, AUC: {best_row['auc']:.4f})",
    )


def run_patch_experiment(
    model_name: str,
    pair_id: str,
    layer: int,
    model: HookedTransformer | None = None,
) -> None:
    """
    Runs the activation patching experiment.

    This function is a wrapper around the `patch_persona_activation` function from `activation_patching.py`.

    Args:
        model_name: The name of the model to use.
        pair_id: The ID of the persona pair to use.
        layer: The layer to perform the patching on.
    """
    if model is None:
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


def run_cot_experiment(
    model_name: str,
    use_hint: bool,
    model: HookedTransformer | None = None,
) -> None:
    """
    Runs the Chain-of-Thought / logit lens experiment.

    This function is a wrapper around the `logit_lens_scores` function from `cot_faithfulness.py`.

    Args:
        model_name: The name of the model to use.
        use_hint: Whether to include the misleading hint in the prompt.
    """
    if model is None:
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
        print(
            f"Final layer logit diff ('{puzzle['choices'][0]}' - '{puzzle['choices'][1]}'): {diffs[-1]:.4f}"
        )


def main() -> None:
    """
    Main function that parses command-line arguments and runs the selected experiments.
    """
    parser = argparse.ArgumentParser(description="Run user identity mechanism interpretability experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["probe", "patch", "cot", "all"],
                        help="Which experiment to run: 'probe', 'patch', 'cot', or 'all'.")
    
    # Arguments for all experiments
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-4b-it",
        help="Model to use (e.g., 'gpt2', 'gpt2-medium').",
    )
    
    # Arguments for the probe experiment
    parser.add_argument("--seed", type=int, default=42, help="Random seed for probe experiment.")
    parser.add_argument("--save_path", type=str, default="probe_results.csv", help="Path to save probe results.")
    parser.add_argument(
        "--n_questions_per_pair",
        type=int,
        default=5,
        help="Number of questions sampled per persona pair.",
    )
    parser.add_argument(
        "--template_holdout",
        action="store_true",
        help="Hold out one prompt template for testing.",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=32,
        help="Maximum number of layers to probe.",
    )
    parser.add_argument(
        "--min_layer",
        type=int,
        default=1,
        help="Minimum layer to probe (use 0 to include layer 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto/cpu/cuda.",
    )
    parser.add_argument(
        "--show_probe_tokens",
        action="store_true",
        help="Print the token that the probe regresses on for a few prompts.",
    )
    parser.add_argument(
        "--show_probe_count",
        type=int,
        default=3,
        help="Number of prompts to show when --show_probe_tokens is set.",
    )
    parser.add_argument(
        "--show_probe_vector",
        action="store_true",
        help="Print the probe vector for a few prompts at a specific layer.",
    )
    parser.add_argument(
        "--show_probe_vector_layer",
        type=int,
        default=1,
        help="Layer to print probe vectors for when --show_probe_vector is set.",
    )
    parser.add_argument(
        "--show_probe_examples",
        action="store_true",
        help="Print a preview of the prompt/label pairs before training.",
    )
    parser.add_argument(
        "--show_probe_examples_count",
        type=int,
        default=5,
        help="Number of prompt/label pairs to show when --show_probe_examples is set.",
    )
    parser.add_argument(
        "--show_embedding_table",
        action="store_true",
        help="Print a table of embedding vectors and labels before training.",
    )
    parser.add_argument(
        "--show_embedding_table_rows",
        type=int,
        default=5,
        help="Number of rows to show in the embedding table.",
    )
    parser.add_argument(
        "--show_embedding_table_dims",
        type=int,
        default=8,
        help="Number of embedding dimensions to show in the table.",
    )
    parser.add_argument(
        "--show_timing",
        action="store_true",
        help="Print timing for each stage of the probe run.",
    )
    parser.add_argument(
        "--probe_position",
        type=str,
        default="question_last_token",
        choices=["persona", "question_end", "question_last_token", "prompt_end"],
        help="Token position to probe.",
    )
    parser.add_argument(
        "--align_persona_lengths",
        action="store_true",
        help="Pad persona strings to equal token length to avoid position leakage.",
    )
    parser.add_argument(
        "--persona_pad_token",
        type=str,
        default=" X",
        help="Token string to use for persona padding when aligned.",
    )
    parser.add_argument(
        "--align_probe_index",
        action="store_true",
        help="Pad persona strings to align the probe token index across classes.",
    )
    parser.add_argument(
        "--probe_template_id",
        type=int,
        default=0,
        help="Template id used to align probe index when --align_probe_index is set.",
    )
    parser.add_argument(
        "--drop_persona",
        action="store_true",
        help="Remove the persona line from prompts as a control.",
    )
    parser.add_argument(
        "--shuffle_labels",
        action="store_true",
        help="Shuffle labels before training as a control.",
    )
    parser.add_argument(
        "--question_holdout",
        action="store_true",
        help="Hold out one question id for testing instead of template holdout.",
    )
    parser.add_argument(
        "--reuse_model",
        action="store_true",
        help="Load the model once and reuse it across experiments.",
    )

    # Arguments for the activation patching experiment
    parser.add_argument("--pair_id", type=str, default="physics", help="Persona pair ID for patching (e.g., 'physics').")
    parser.add_argument("--layer", type=int, default=5, help="Layer to use for activation patching.")

    # Arguments for the CoT experiment
    parser.add_argument("--use_hint", action="store_true", help="Use misleading hint in the CoT puzzle.")
    
    args = parser.parse_args()

    print(f"Running experiment(s): '{args.experiment}' with model '{args.model_name}'")

    shared_model = None
    if args.reuse_model and args.experiment in ["probe", "patch", "cot", "all"]:
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        device_str = str(device)
        if device_str.isdigit():
            device_str = f"cuda:{device_str}"
        if device_str.startswith("cuda"):
            try:
                torch.cuda.set_device(0)
            except Exception:
                device_str = "cuda:0"
        shared_model = HookedTransformer.from_pretrained(args.model_name, device=device_str)

    if args.experiment in ["probe", "all"]:
        run_probe_experiment(
            args.model_name,
            args.seed,
            args.save_path,
            args.n_questions_per_pair,
            args.template_holdout,
            args.max_layers,
            args.device,
            args.min_layer,
            args.show_probe_tokens,
            args.show_probe_count,
            args.show_probe_vector,
            args.show_probe_vector_layer,
            args.show_probe_examples,
            args.show_probe_examples_count,
            args.show_embedding_table,
            args.show_embedding_table_rows,
            args.show_embedding_table_dims,
            args.show_timing,
            args.probe_position,
            args.align_persona_lengths,
            args.persona_pad_token,
            args.align_probe_index,
            args.probe_template_id,
            args.drop_persona,
            args.shuffle_labels,
            args.question_holdout,
            model=shared_model,
        )

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
        
        run_patch_experiment(args.model_name, args.pair_id, layer_for_patching, model=shared_model)

    if args.experiment in ["cot", "all"]:
        run_cot_experiment(args.model_name, args.use_hint, model=shared_model)


if __name__ == "__main__":
    main()
