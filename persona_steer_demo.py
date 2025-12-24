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


def build_neutral_prompt(question: str, template_id: int = 0, drop_persona: bool = False) -> str:
    template = PROMPT_TEMPLATES[template_id]
    prompt = template.format(
        persona_marker=PERSONA_MARKER,
        persona="",
        question=question,
    )
    if drop_persona:
        return prompt.replace(f"{PERSONA_MARKER}\nPersona: \n", "")
    return prompt


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


def kl_scores(
    model: HookedTransformer,
    prompts: List[str],
    direction: np.ndarray,
    layer: int,
    alpha: float,
) -> float:
    lengths = [model.to_tokens(p)[0].shape[0] for p in prompts]
    tokens = model.to_tokens(prompts)
    hook_name = f"blocks.{layer}.hook_resid_pre"
    direction_t = torch.tensor(direction, device=model.cfg.device, dtype=torch.float32)

    def steer_hook(activation, hook):
        return activation + alpha * direction_t

    _, base_cache = model.run_with_cache(tokens)
    if alpha != 0.0:
        try:
            with model.hooks(fwd_hooks=[(hook_name, steer_hook)]):
                _, steer_cache = model.run_with_cache(tokens)
        except TypeError:
            _, steer_cache = model.run_with_cache(tokens, hooks=[(hook_name, steer_hook)])
    else:
        steer_cache = base_cache

    kls = []
    for i, length in enumerate(lengths):
        base_resid = base_cache["resid_pre", layer][i, length - 1]
        steer_resid = steer_cache["resid_pre", layer][i, length - 1]
        base_logits = model.unembed(base_resid)
        steer_logits = model.unembed(steer_resid)
        base_probs = torch.softmax(base_logits, dim=0)
        steer_probs = torch.softmax(steer_logits, dim=0)
        kl = torch.sum(base_probs * (torch.log(base_probs + 1e-12) - torch.log(steer_probs + 1e-12)))
        kls.append(float(kl.item()))
    return float(np.mean(kls))


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
    parser.add_argument("--drop_persona", action="store_true", help="Remove persona line from prompts.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling for generation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_direction", action="store_true", help="Use a random direction baseline.")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--kl_report", action="store_true", help="Compute KL(base||steered) per layer/alpha.")
    parser.add_argument("--kl_out", type=str, default="report_artifacts/steer_kl.csv")
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
    if args.random_direction:
        rng = np.random.default_rng(args.random_seed)
        random_dir = rng.standard_normal(direction.shape)
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8)
        direction = random_dir * (np.linalg.norm(direction) + 1e-8)
    direction_t = torch.tensor(direction, device=model.cfg.device, dtype=torch.float32)
    torch.manual_seed(args.seed)

    use_sampling = args.do_sample or args.temperature > 0.0 or args.top_p < 1.0
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": use_sampling,
    }
    if use_sampling:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    outputs: List[Dict[str, str]] = []
    for q in prompts:
        prompt = build_neutral_prompt(q, args.template_id, drop_persona=args.drop_persona)
        tokens = model.to_tokens(prompt)
        clean_tokens = model.generate(tokens, **gen_kwargs)
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
                        steered_tokens = model.generate(tokens, **gen_kwargs)
                except TypeError:
                    steered_tokens = model.generate(
                        tokens,
                        **gen_kwargs,
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

    if args.kl_report:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        prompt_texts = [
            build_neutral_prompt(q, args.template_id, drop_persona=args.drop_persona)
            for q in prompts
        ]
        rows = ["layer,alpha,kl_base_steered"]
        for layer in layers:
            for alpha in alphas:
                kl_val = kl_scores(model, prompt_texts, direction, layer, alpha)
                rows.append(f"{layer},{alpha},{kl_val}")
        Path(args.kl_out).write_text("\n".join(rows))
        print("Saved KL summary:", args.kl_out)

    if plt is not None:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        prompt_texts = [
            build_neutral_prompt(q, args.template_id, drop_persona=args.drop_persona)
            for q in prompts
        ]
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
