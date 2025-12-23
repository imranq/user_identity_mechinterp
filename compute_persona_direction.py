import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer

from persona_prompts import build_prompt_dataset
from probe_user_representation import get_probe_token_index


def extract_activations(
    model: HookedTransformer,
    prompts: List[Tuple[str, int]],
    layer: int,
    probe_position: str,
) -> Tuple[np.ndarray, np.ndarray]:
    activations = []
    labels = []
    for prompt, label in prompts:
        token_index = get_probe_token_index(model, prompt, probe_position)
        _, cache = model.run_with_cache(prompt)
        resid = cache["resid_pre", layer][0, token_index].detach().cpu().numpy()
        activations.append(resid)
        labels.append(label)
    return np.vstack(activations), np.array(labels)


def mean_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mean_expert = X[y == 0].mean(axis=0)
    mean_novice = X[y == 1].mean(axis=0)
    return mean_expert - mean_novice


def probe_direction(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    clf = LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed)
    clf.fit(X, y)
    return clf.coef_[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute persona direction")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument(
        "--probe_position",
        type=str,
        default="question_last_token",
        choices=["persona", "question_end", "question_last_token", "prompt_end"],
    )
    parser.add_argument("--n_questions_per_pair", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--method",
        type=str,
        default="mean",
        choices=["mean", "probe"],
        help="How to compute the direction: mean-diff or probe weights.",
    )
    parser.add_argument("--align_probe_index", action="store_true")
    parser.add_argument("--probe_template_id", type=int, default=0)
    parser.add_argument("--persona_pad_token", type=str, default=" X")
    parser.add_argument("--save_path", type=str, default="persona_direction.npy")
    parser.add_argument("--meta_path", type=str, default="persona_direction.json")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    prompt_objs = build_prompt_dataset(
        n_questions_per_pair=args.n_questions_per_pair,
        seed=args.seed,
        align_persona_lengths=False,
        tokenizer=model if args.align_probe_index else None,
        pad_token=args.persona_pad_token,
        align_probe_index=args.align_probe_index,
        probe_template_id=args.probe_template_id,
        drop_persona=False,
    )
    prompts = [(p.prompt, p.label) for p in prompt_objs]

    X, y = extract_activations(model, prompts, args.layer, args.probe_position)
    if args.method == "mean":
        direction = mean_direction(X, y)
    else:
        direction = probe_direction(X, y, args.seed)

    direction = direction / (np.linalg.norm(direction) + 1e-8)
    np.save(args.save_path, direction)

    meta = {
        "model_name": args.model_name,
        "layer": args.layer,
        "probe_position": args.probe_position,
        "method": args.method,
        "n_examples": len(prompts),
        "align_probe_index": args.align_probe_index,
        "probe_template_id": args.probe_template_id,
        "persona_pad_token": args.persona_pad_token,
    }
    Path(args.meta_path).write_text(json.dumps(meta, indent=2))

    print("Saved direction to", args.save_path)
    print("Saved metadata to", args.meta_path)


if __name__ == "__main__":
    main()
