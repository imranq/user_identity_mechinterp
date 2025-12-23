#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformer_lens import HookedTransformer

from persona_prompts import PROMPT_TEMPLATES, PERSONA_MARKER

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

DEFAULT_PROMPTS = [
    "Explain how a refrigerator works.",
    "Write a short story about a lost kite.",
    "Describe how rainbows form.",
    "Give advice for studying effectively.",
    "Explain the basics of chess.",
]


def build_neutral_prompt(question: str, template_id: int = 0) -> str:
    template = PROMPT_TEMPLATES[template_id]
    return template.format(
        persona_marker=PERSONA_MARKER,
        persona="",
        question=question,
    ).replace(f"{PERSONA_MARKER}\nPersona: \n", "")


def projection_scores(
    model: HookedTransformer,
    prompts: List[str],
    direction: np.ndarray,
    layer: int,
    alpha: float,
) -> List[float]:
    direction_t = torch.tensor(direction, device=model.cfg.device, dtype=torch.float32)
    tokens = model.to_tokens(prompts)
    lengths = [model.to_tokens(p)[0].shape[0] for p in prompts]
    hook_name = f"blocks.{layer}.hook_resid_pre"

    def steer_hook(activation, hook):
        return activation + alpha * direction_t

    if alpha != 0.0:
        try:
            with model.hooks(fwd_hooks=[(hook_name, steer_hook)]):
                _, cache = model.run_with_cache(tokens)
        except TypeError:
            _, cache = model.run_with_cache(tokens, hooks=[(hook_name, steer_hook)])
    else:
        _, cache = model.run_with_cache(tokens)

    resid = cache["resid_pre", layer]
    scores: List[float] = []
    for i, length in enumerate(lengths):
        vec = resid[i, length - 1]
        scores.append(float(torch.dot(vec, direction_t).item()))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona steering demo.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--direction_path", type=str, default="persona_direction.npy")
    parser.add_argument("--layers", type=str, default="4", help="Comma-separated layer list.")
    parser.add_argument("--alphas", type=str, default="8.0", help="Comma-separated alpha list.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--template_id", type=int, default=0)
    parser.add_argument("--prompts_path", type=str, default="")
    parser.add_argument("--out_path", type=str, default="persona_steer_outputs.jsonl")
    parser.add_argument("--plot_dir", type=str, default="report_artifacts")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    prompts = DEFAULT_PROMPTS
    if args.prompts_path:
        prompts = Path(args.prompts_path).read_text().strip().splitlines()
        prompts = [p for p in prompts if p.strip()]

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    direction = np.load(args.direction_path)
    direction_t = torch.tensor(direction, device=model.cfg.device, dtype=torch.float32)

    outputs: List[Dict[str, str]] = []
    for q in prompts:
        prompt = build_neutral_prompt(q, args.template_id)
        tokens = model.to_tokens(prompt)
        clean_tokens = model.generate(tokens, max_new_tokens=args.max_new_tokens, do_sample=False)
        clean_text = model.to_string(clean_tokens[0])

        print("\n=== Question ===")
        print(q)
        print("\n--- Clean ---")
        print(clean_text)

        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_pre"
            for alpha in alphas:
                def steer_hook(activation, hook):
                    activation[:, :] = activation + alpha * direction_t
                    return activation

                try:
                    with model.hooks(fwd_hooks=[(hook_name, steer_hook)]):
                        steered_tokens = model.generate(
                            tokens, max_new_tokens=args.max_new_tokens, do_sample=False
                        )
                except TypeError:
                    steered_tokens = model.generate(
                        tokens,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        hooks=[(hook_name, steer_hook)],
                    )
                steered_text = model.to_string(steered_tokens[0])
                outputs.append(
                    {
                        "question": q,
                        "prompt": prompt,
                        "clean_output": clean_text,
                        "steered_output": steered_text,
                        "alpha": alpha,
                        "layer": layer,
                    }
                )
                print(f"\n--- Steered (layer={layer}, alpha={alpha}) ---")
                print(steered_text)

    if plt is not None:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        prompt_texts = [build_neutral_prompt(q, args.template_id) for q in prompts]
        for layer in layers:
            clean_scores = projection_scores(model, prompt_texts, direction, layer, 0.0)
            for alpha in alphas:
                steered_scores = projection_scores(model, prompt_texts, direction, layer, alpha)
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(clean_scores, label="clean", marker="o")
                ax.plot(steered_scores, label=f"steered α={alpha}", marker="o")
                ax.set_xlabel("prompt index")
                ax.set_ylabel("direction projection")
                ax.set_title(f"Layer {layer} projection shift")
                ax.legend()
                fig.tight_layout()
                safe_alpha = str(alpha).replace(".", "p").replace("-", "m")
                out_path = plot_dir / f"steer_proj_layer{layer}_alpha{safe_alpha}.png"
                fig.savefig(out_path)
                plt.close(fig)
                print("Saved plot:", out_path)

    Path(args.out_path).write_text("\n".join(json.dumps(o) for o in outputs))
    print("\nWrote:", args.out_path)


if __name__ == "__main__":
    main()
