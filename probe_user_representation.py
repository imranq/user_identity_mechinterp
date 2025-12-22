"""
This script trains a linear probe to investigate where in a transformer model persona information is represented.
A linear probe is a simple machine learning model (in this case, logistic regression) trained to predict a
property of interest (here, the persona) from the model's internal activations.

The process is as follows:
1. For each persona prompt, extract the hidden state (activation) from a specific layer at the token
   position corresponding to the persona.
2. Train a logistic regression classifier on these activations to predict the persona label (e.g., expert vs. novice).
3. Repeat this for every layer in the model.
4. The accuracy of the probe at each layer indicates how linearly separable the persona representations are
   at that layer. High accuracy suggests that the persona information is strongly and accessibly encoded.
"""

import argparse
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformer_lens import HookedTransformer

from persona_prompts import build_prompts, PERSONA_MARKER


def find_subsequence(sequence: List[int], subsequence: List[int]) -> int:
    """
    Finds the starting index of a subsequence within a sequence.

    Args:
        sequence: The main list of integers.
        subsequence: The sublist of integers to find.

    Returns:
        The starting index of the subsequence, or -1 if not found.
    """
    for idx in range(len(sequence) - len(subsequence) + 1):
        if sequence[idx:idx + len(subsequence)] == subsequence:
            return idx
    return -1


def get_persona_token_index(model: HookedTransformer, prompt: str) -> int:
    """
    Finds the index of the token immediately following the PERSONA_MARKER in a prompt.

    This token's activation is used as the representation of the persona for probing.

    Args:
        model: The HookedTransformer model, used for tokenization.
        prompt: The prompt string.

    Returns:
        The index of the first token of the persona description.
    """
    tokens = model.to_tokens(prompt)[0].tolist()
    # We look for the marker without the initial BOS token
    marker_tokens = model.to_tokens(PERSONA_MARKER, prepend_bos=False)[0].tolist()
    marker_start = find_subsequence(tokens, marker_tokens)
    if marker_start == -1:
        raise ValueError("Persona marker not found in token sequence")
    # Return the index of the token immediately after the marker.
    return marker_start + len(marker_tokens)


def extract_layer_activations(
    model: HookedTransformer,
    prompts: List[Tuple[str, int]],
    layer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts activations from a specific layer for a list of prompts.

    Args:
        model: The HookedTransformer model.
        prompts: A list of tuples, where each tuple is (prompt_string, label).
        layer: The layer from which to extract activations.

    Returns:
        A tuple containing:
        - A numpy array of the activations (num_prompts, hidden_size).
        - A numpy array of the corresponding labels.
    """
    activations = []
    labels = []
    for prompt, label in prompts:
        token_index = get_persona_token_index(model, prompt)
        # Run the model and cache activations.
        _, cache = model.run_with_cache(prompt)
        # Get the residual stream activation at the persona token index for the specified layer.
        resid = cache["resid_pre", layer][0, token_index].detach().cpu().numpy()
        activations.append(resid)
        labels.append(label)
    return np.vstack(activations), np.array(labels)


def run_probe(model_name: str, seed: int) -> pd.DataFrame:
    """
    Runs the linear probing experiment across all layers of a model.

    Args:
        model_name: The name of the transformer model to load.
        seed: A random seed for reproducibility.

    Returns:
        A pandas DataFrame with the probing accuracy for each layer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    prompt_objs = build_prompts()
    prompts = [(p.prompt, p.label) for p in prompt_objs]

    results = []
    # Iterate through each layer of the model.
    for layer in range(model.cfg.n_layers):
        # Extract activations for the current layer.
        X, y = extract_layer_activations(model, prompts, layer)
        # Split data into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
        # Train a logistic regression classifier.
        clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=seed)
        clf.fit(X_train, y_train)
        # Evaluate the classifier on the test set.
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({"layer": layer, "accuracy": acc})

    df = pd.DataFrame(results)
    return df


def main() -> None:
    """
    Main function to run the probing experiment.
    Parses command-line arguments, runs the probe, saves the results, and prints the best layer.
    """
    parser = argparse.ArgumentParser(description="Probe persona representations")
    parser.add_argument("--model_name", type=str, default="gpt2", help="The name of the model to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save_path", type=str, default="probe_results.csv", help="Path to save the results CSV.")
    args = parser.parse_args()

    # Run the probing experiment.
    df = run_probe(args.model_name, args.seed)
    # Save the results to a CSV file.
    df.to_csv(args.save_path, index=False)

    # Find and print the layer with the highest accuracy.
    best_row = df.loc[df["accuracy"].idxmax()]
    print("Probe results saved to", args.save_path)
    print("Best layer:", int(best_row["layer"]), "accuracy:", float(best_row["accuracy"]))


if __name__ == "__main__":
    main()
