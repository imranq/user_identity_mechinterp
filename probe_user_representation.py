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
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformer_lens import HookedTransformer

from persona_prompts import build_prompt_dataset, PERSONA_MARKER


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
    max_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts activations for all layers up to max_layers for a list of prompts.

    Args:
        model: The HookedTransformer model.
        prompts: A list of tuples, where each tuple is (prompt_string, label).
        max_layers: The number of layers to extract activations for.

    Returns:
        A tuple containing:
        - A numpy array of the activations (num_prompts, num_layers, hidden_size).
        - A numpy array of the corresponding labels.
    """
    activations = []
    labels = []
    for prompt, label in tqdm(prompts, desc="Extracting activations", unit="prompt"):
        token_index = get_persona_token_index(model, prompt)
        _, cache = model.run_with_cache(prompt)
        layer_acts = []
        for layer in range(max_layers):
            resid = cache["resid_pre", layer][0, token_index].detach().cpu().numpy()
            layer_acts.append(resid)
        activations.append(np.stack(layer_acts))
        labels.append(label)
    return np.stack(activations), np.array(labels)


def run_probe(
    model_name: str,
    seed: int,
    n_questions_per_pair: int,
    template_holdout: bool,
    max_layers: int,
    device: str,
    min_layer: int,
) -> pd.DataFrame:
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
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    prompt_objs = build_prompt_dataset(
        n_questions_per_pair=n_questions_per_pair,
        seed=seed,
    )
    prompts = [(p.prompt, p.label) for p in prompt_objs]
    template_ids = [p.template_id for p in prompt_objs]

    if template_holdout:
        unique_templates = sorted(set(template_ids))
        test_template = unique_templates[-1]
        train_indices = [i for i, t_id in enumerate(template_ids) if t_id != test_template]
        test_indices = [i for i, t_id in enumerate(template_ids) if t_id == test_template]
    else:
        train_indices, test_indices = None, None

    n_layers = min(model.cfg.n_layers, max_layers)
    X_all, y = extract_layer_activations(model, prompts, n_layers)

    results = []
    start_layer = max(0, min_layer)
    for layer in tqdm(range(start_layer, n_layers), desc="Training probes", unit="layer"):
        X = X_all[:, layer, :]
        if template_holdout:
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]
        else:
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
        if len(set(y_test)) > 1:
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = float("nan")
        results.append({"layer": layer, "accuracy": acc, "auc": auc})

    df = pd.DataFrame(results)
    return df


def main() -> None:
    """
    Main function to run the probing experiment.
    Parses command-line arguments, runs the probe, saves the results, and prints the best layer.
    """
    parser = argparse.ArgumentParser(description="Probe persona representations")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-4b-it",
        help="The name of the model to use.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
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
    parser.add_argument("--save_path", type=str, default="probe_results.csv", help="Path to save the results CSV.")
    args = parser.parse_args()

    # Run the probing experiment.
    df = run_probe(
        args.model_name,
        args.seed,
        args.n_questions_per_pair,
        args.template_holdout,
        args.max_layers,
        args.device,
        args.min_layer,
    )
    # Save the results to a CSV file.
    df.to_csv(args.save_path, index=False)

    # Find and print the layer with the highest accuracy.
    best_row = df.loc[df["accuracy"].idxmax()]
    print("Probe results saved to", args.save_path)
    print(
        "Best layer:",
        int(best_row["layer"]),
        "accuracy:",
        float(best_row["accuracy"]),
        "auc:",
        float(best_row["auc"]),
    )


if __name__ == "__main__":
    main()
