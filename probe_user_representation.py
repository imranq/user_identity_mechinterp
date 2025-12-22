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
    for idx in range(len(sequence) - len(subsequence) + 1):
        if sequence[idx:idx + len(subsequence)] == subsequence:
            return idx
    return -1


def get_persona_token_index(model: HookedTransformer, prompt: str) -> int:
    tokens = model.to_tokens(prompt)[0].tolist()
    marker_tokens = model.to_tokens(PERSONA_MARKER)[0].tolist()
    marker_start = find_subsequence(tokens, marker_tokens)
    if marker_start == -1:
        raise ValueError("Persona marker not found in token sequence")
    return marker_start + len(marker_tokens)


def extract_layer_activations(
    model: HookedTransformer,
    prompts: List[str],
    layer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    activations = []
    labels = []
    for prompt, label in prompts:
        token_index = get_persona_token_index(model, prompt)
        _, cache = model.run_with_cache(prompt)
        resid = cache["resid_pre", layer][0, token_index].detach().cpu().numpy()
        activations.append(resid)
        labels.append(label)
    return np.vstack(activations), np.array(labels)


def run_probe(model_name: str, seed: int) -> pd.DataFrame:
    torch.manual_seed(seed)
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    prompt_objs = build_prompts()
    prompts = [(p.prompt, p.label) for p in prompt_objs]

    results = []
    for layer in range(model.cfg.n_layers):
        X, y = extract_layer_activations(model, prompts, layer)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=1000, n_jobs=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({"layer": layer, "accuracy": acc})

    df = pd.DataFrame(results)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe persona representations")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="probe_results.csv")
    args = parser.parse_args()

    df = run_probe(args.model_name, args.seed)
    df.to_csv(args.save_path, index=False)

    best_row = df.loc[df["accuracy"].idxmax()]
    print("Probe results saved to", args.save_path)
    print("Best layer:", int(best_row["layer"]), "accuracy:", float(best_row["accuracy"]))


if __name__ == "__main__":
    main()
