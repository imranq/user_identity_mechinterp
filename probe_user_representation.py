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
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, roc_curve
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


def get_probe_token_index(
    model: HookedTransformer,
    prompt: str,
    position: str,
) -> int:
    if position == "persona":
        return get_persona_token_index(model, prompt)
    if position == "prompt_end":
        tokens = model.to_tokens(prompt)[0].tolist()
        return len(tokens) - 1
    if position == "question_end":
        prefix, sep, _ = prompt.rpartition("Answer:")
        if not sep:
            raise ValueError("Prompt does not contain 'Answer:' marker")
        prefix_tokens = model.to_tokens(prefix)[0].tolist()
        if len(prefix_tokens) == 0:
            raise ValueError("Question prefix tokenization is empty")
        return len(prefix_tokens) - 1
    if position == "question_last_token":
        prefix, sep, _ = prompt.rpartition("Answer:")
        if not sep:
            raise ValueError("Prompt does not contain 'Answer:' marker")
        prefix_str_tokens = model.to_str_tokens(prefix)
        if len(prefix_str_tokens) == 0:
            raise ValueError("Question prefix tokenization is empty")
        for idx in range(len(prefix_str_tokens) - 1, -1, -1):
            if prefix_str_tokens[idx].strip() != "":
                return idx
        raise ValueError("No non-whitespace token found in question prefix")
    raise ValueError(f"Unknown probe position: {position}")

def extract_layer_activations(
    model: HookedTransformer,
    prompts: List[Tuple[str, int]],
    max_layers: int,
    show_tokens: bool,
    show_count: int,
    show_vector: bool,
    show_vector_layer: int,
    probe_position: str,
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
    for idx, (prompt, label) in enumerate(
        tqdm(prompts, desc="Extracting activations", unit="prompt")
    ):
        token_index = get_probe_token_index(model, prompt, probe_position)
        if show_tokens and idx < show_count:
            tokens = model.to_str_tokens(prompt)
            marker_tokens = model.to_str_tokens(PERSONA_MARKER, prepend_bos=False)
            print("\n--- Probe token inspection ---")
            print("Prompt index:", idx)
            print("Label:", label)
            print("Probe position:", probe_position)
            print("Marker tokens:", marker_tokens)
            print("Probe token index:", token_index)
            print("Probe token string:", tokens[token_index])
        _, cache = model.run_with_cache(prompt)
        layer_acts = []
        for layer in range(max_layers):
            resid = cache["resid_pre", layer][0, token_index].detach().cpu().numpy()
            if show_vector and idx < show_count and layer == show_vector_layer:
                print("\n--- Probe vector inspection ---")
                print("Prompt index:", idx)
                print("Layer:", layer)
                print("Vector shape:", resid.shape)
                print("Vector L2 norm:", float(np.linalg.norm(resid)))
                print("Vector head (8):", np.array2string(resid[:8], precision=4))
            layer_acts.append(resid)
        activations.append(np.stack(layer_acts))
        labels.append(label)
    return np.stack(activations), np.array(labels)


def run_probe(
    model_name: str,
    seed: int,
    n_questions_per_pair: int,
    template_holdout: bool,
    question_holdout: bool,
    max_layers: int,
    device: str,
    min_layer: int,
    show_tokens: bool,
    show_count: int,
    show_vector: bool,
    show_vector_layer: int,
    show_examples: bool,
    show_examples_count: int,
    show_embedding_table: bool,
    show_embedding_table_rows: int,
    show_embedding_table_dims: int,
    show_timing: bool,
    probe_position: str,
    align_persona_lengths: bool,
    pad_token: str,
    align_probe_index: bool,
    probe_template_id: int,
    drop_persona: bool,
    shuffle_labels: bool,
    model: Optional[HookedTransformer] = None,
) -> pd.DataFrame:
    start_time = time.perf_counter()
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
    if model is None:
        t0 = time.perf_counter()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = HookedTransformer.from_pretrained(model_name, device=device)
        if show_timing:
            print(f"Timing: model load {time.perf_counter() - t0:.2f}s")
        device_name = device
        display_name = model_name
    else:
        device_name = str(model.cfg.device)
        display_name = getattr(model.cfg, "model_name", model_name)
    t1 = time.perf_counter()
    prompt_objs = build_prompt_dataset(
        n_questions_per_pair=n_questions_per_pair,
        seed=seed,
        align_persona_lengths=align_persona_lengths,
        tokenizer=model if (align_persona_lengths or align_probe_index) else None,
        pad_token=pad_token,
        align_probe_index=align_probe_index,
        probe_template_id=probe_template_id,
        drop_persona=drop_persona,
    )
    if show_timing:
        print(f"Timing: dataset build {time.perf_counter() - t1:.2f}s")
    prompts = [(p.prompt, p.label) for p in prompt_objs]
    template_ids = [p.template_id for p in prompt_objs]
    pair_ids = sorted({p.pair_id for p in prompt_objs})
    unique_templates = sorted(set(template_ids))
    question_ids = [p.question_id for p in prompt_objs]
    unique_question_ids = sorted(set(question_ids))

    print("\n--- Probe configuration ---")
    print("Model:", display_name)
    print("Device:", device_name)
    print("Examples:", len(prompts))
    print("Pairs:", ", ".join(pair_ids))
    print("Questions per pair:", n_questions_per_pair)
    print("Unique questions:", len(unique_question_ids))
    print("Templates:", unique_templates)
    print("Template holdout:", template_holdout)
    print("Probe position:", probe_position)
    print("Align persona lengths:", align_persona_lengths)
    if align_persona_lengths:
        print("Persona pad token:", repr(pad_token))
    print("Align probe index:", align_probe_index)
    if align_probe_index:
        print("Probe template id:", probe_template_id)
    print("Drop persona:", drop_persona)
    print("Shuffle labels:", shuffle_labels)
    print("Question holdout:", question_holdout)
    print("Layer range:", f"{max(0, min_layer)}..{min(model.cfg.n_layers - 1, max_layers - 1)}")

    if show_examples:
        print("\n--- Probe dataset preview ---")
        for idx, (prompt, label) in enumerate(prompts[:show_examples_count]):
            preview = prompt.replace("\n", "\\n")
            if len(preview) > 240:
                preview = preview[:240] + "..."
            print(f"[{idx}] label={label} prompt='{preview}'")

    if question_holdout:
        test_question = unique_question_ids[-1]
        train_indices = [i for i, q_id in enumerate(question_ids) if q_id != test_question]
        test_indices = [i for i, q_id in enumerate(question_ids) if q_id == test_question]
    elif template_holdout:
        test_template = unique_templates[-1]
        train_indices = [i for i, t_id in enumerate(template_ids) if t_id != test_template]
        test_indices = [i for i, t_id in enumerate(template_ids) if t_id == test_template]
    else:
        train_indices, test_indices = None, None

    n_layers = min(model.cfg.n_layers, max_layers)
    t2 = time.perf_counter()
    X_all, y = extract_layer_activations(
        model,
        prompts,
        n_layers,
        show_tokens,
        show_count,
        show_vector,
        show_vector_layer,
        probe_position,
    )
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        rng.shuffle(y)
    if show_timing:
        print(f"Timing: activation extraction {time.perf_counter() - t2:.2f}s")

    if show_embedding_table:
        layer_idx = max(0, min(show_vector_layer, n_layers - 1))
        rows = min(show_embedding_table_rows, X_all.shape[0])
        dims = min(show_embedding_table_dims, X_all.shape[2])
        table = pd.DataFrame(X_all[:rows, layer_idx, :dims])
        table.insert(0, "label", y[:rows])
        print("\n--- Probe embedding table ---")
        print(f"Layer: {layer_idx} | Rows: {rows} | Dims: {dims}")
        print(table.to_string(index=False))

    results = []
    start_layer = max(0, min_layer)
    t3 = time.perf_counter()
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
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            balanced_scores = (tpr + (1 - fpr)) / 2
            best_idx = int(np.argmax(balanced_scores))
            best_thresh = float(thresholds[best_idx])
            bal_acc = float(balanced_scores[best_idx])
        else:
            y_prob = None
            auc = float("nan")
            bal_acc = float("nan")
            best_thresh = float("nan")
        results.append(
            {
                "layer": layer,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "auc": auc,
                "best_threshold": best_thresh,
            }
        )
    if show_timing:
        print(f"Timing: probe training {time.perf_counter() - t3:.2f}s")
        print(f"Timing: total run {time.perf_counter() - start_time:.2f}s")

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
    parser.add_argument("--save_path", type=str, default="probe_results.csv", help="Path to save the results CSV.")
    args = parser.parse_args()

    # Run the probing experiment.
    df = run_probe(
        args.model_name,
        args.seed,
        args.n_questions_per_pair,
        args.template_holdout,
        args.question_holdout,
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
