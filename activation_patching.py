
"""
This script performs activation patching on a transformer model to analyze the effect of persona-specific activations.
Activation patching involves running the model on a "source" prompt, caching an activation from a specific token
(in this case, a persona-defining token), and then running the model on a "target" prompt while substituting
the cached activation at the corresponding token position.

The goal is to see if the persona activation from the source prompt can influence the stylistic output of the
model on the target prompt. The script measures this by comparing the logit difference between a "formal" and
a "simple" style token before and after patching.
"""

import argparse
from typing import Dict

import torch
from transformer_lens import HookedTransformer

from persona_prompts import build_prompts, PERSONA_MARKER
from probe_user_representation import get_persona_token_index


# A dictionary mapping style names to representative tokens.
# These tokens are used to measure the model's stylistic output.
STYLE_TOKENS = {
    "formal": "Therefore",
    "simple": "So",
}


def get_style_logit_diff(model: HookedTransformer, prompt: str) -> float:
    """
    Calculates the difference in logits between the 'formal' and 'simple' style tokens for a given prompt.

    Args:
        model: The HookedTransformer model.
        prompt: The input prompt string.

    Returns:
        The difference between the logit for the 'formal' token and the logit for the 'simple' token.
    """
    tokens = model.to_tokens(prompt)
    # Get the model's output logits for the last token in the prompt.
    logits = model(tokens)[0, -1]
    # Get the token IDs for our style indicator tokens.
    formal_id = model.to_tokens(STYLE_TOKENS["formal"])[0, 0].item()
    simple_id = model.to_tokens(STYLE_TOKENS["simple"])[0, 0].item()
    # Return the difference in logits between the two style tokens.
    return (logits[formal_id] - logits[simple_id]).item()


def patch_persona_activation(
    model: HookedTransformer,
    source_prompt: str,
    target_prompt: str,
    layer: int,
) -> Dict[str, float]:
    """
    Patches the persona activation from a source prompt into a target prompt and measures the effect on style.

    This function runs the model on the `source_prompt`, caches the residual stream activation at the
    persona token, and then runs the model on the `target_prompt` with a hook that replaces the
    activation at the target's persona token with the cached activation from the source.

    Args:
        model: The HookedTransformer model.
        source_prompt: The prompt from which to take the activation.
        target_prompt: The prompt to patch the activation into.
        layer: The layer from which to take the residual stream activation.

    Returns:
        A dictionary containing the style logit difference for the clean run, the patched run, and the delta.
    """
    # Find the index of the persona marker token in both prompts.
    source_index = get_persona_token_index(model, source_prompt)
    target_index = get_persona_token_index(model, target_prompt)

    # Run the model on the source prompt and cache the activations.
    _, source_cache = model.run_with_cache(source_prompt)
    # Get the specific activation we want to patch from the cache.
    source_resid = source_cache["resid_pre", layer][0, source_index].detach().clone()

    def patch_hook(activation, hook):
        """A hook function that replaces the activation at the target index."""
        activation[0, target_index] = source_resid
        return activation

    # Get the baseline style score for the target prompt without any patching.
    clean_score = get_style_logit_diff(model, target_prompt)
    # Run the model with the patching hook.
    patched_logits = model.run_with_hooks(
        target_prompt,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook)],
    )
    # Calculate the style score from the patched run.
    patched_score = (patched_logits[0, -1, model.to_tokens(STYLE_TOKENS["formal"])[0, 0]] -
                     patched_logits[0, -1, model.to_tokens(STYLE_TOKENS["simple"])[0, 0]]).item()

    # Return the results.
    return {
        "clean_score": clean_score,
        "patched_score": patched_score,
        "delta": patched_score - clean_score,
    }


def main() -> None:
    """
    Main function to run the activation patching experiment.
    Parses command-line arguments to select the model, persona pair, and layer for the experiment.
    It then loads the model, finds the relevant prompts, performs the activation patching,
    and prints the results.
    """
    parser = argparse.ArgumentParser(description="Activation patching for persona tokens")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-4b-it",
        help="The name of the model to use (e.g., 'gpt2').",
    )
    parser.add_argument("--pair_id", type=str, default="physics", help="The ID of the persona pair to use (e.g., 'physics').")
    parser.add_argument("--layer", type=int, default=5, help="The layer to perform the activation patching on.")
    args = parser.parse_args()

    # Load the pre-trained transformer model.
    model = HookedTransformer.from_pretrained(args.model_name, device="cpu")
    # Filter the prompts to get the pair specified by the command-line argument.
    prompts = [p for p in build_prompts() if p.pair_id == args.pair_id]
    if len(prompts) != 2:
        raise ValueError("pair_id not found or not exactly two prompts")

    # Separate the expert and novice prompts.
    expert_prompt = next(p.prompt for p in prompts if p.role == "expert")
    novice_prompt = next(p.prompt for p in prompts if p.role == "novice")

    # Perform the activation patching.
    results = patch_persona_activation(
        model=model,
        source_prompt=expert_prompt,
        target_prompt=novice_prompt,
        layer=args.layer,
    )

    # Print the results.
    print("Activation patching results")
    print("Pair:", args.pair_id)
    print("Layer:", args.layer)
    print("Clean logit diff:", results["clean_score"])
    print("Patched logit diff:", results["patched_score"])
    print("Delta:", results["delta"])


if __name__ == "__main__":
    main()
