#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformer_lens import HookedTransformer

from persona_prompts import build_prompt_dataset
from persona_patching_runner import apply_direction
from probe_user_representation import get_probe_token_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy patching demo for a few prompts.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--direction_path", type=str, default="persona_direction.npy")
    parser.add_argument("--layers", type=str, default="4", help="Comma-separated layer list.")
    parser.add_argument("--alphas", type=str, default="3.0", help="Comma-separated alpha list.")
    parser.add_argument("--n_questions_per_pair", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--probe_position", type=str, default="question_last_token")
    parser.add_argument("--align_probe_index", action="store_true")
    parser.add_argument("--probe_template_id", type=int, default=0)
    parser.add_argument("--persona_pad_token", type=str, default=" X")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--max_prompts", type=int, default=3)
    parser.add_argument("--out_path", type=str, default="toy_patch_outputs.jsonl")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    print("Model:", args.model_name)
    print("Device:", device)
    print("Layers:", layers)
    print("Alphas:", alphas)
    print("Max prompts:", args.max_prompts)
    print("Max new tokens:", args.max_new_tokens)

    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    direction = np.load(args.direction_path)

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
    prompt_objs = prompt_objs[: args.max_prompts]
    print("Prompts:", len(prompt_objs))

    outputs: List[Dict[str, str]] = []
    for idx, item in enumerate(prompt_objs):
        tokens = model.to_tokens(item.prompt)
        clean_tokens = model.generate(tokens, max_new_tokens=args.max_new_tokens, do_sample=False)
        clean_text = model.to_string(clean_tokens[0])
        token_index = get_probe_token_index(model, item.prompt, args.probe_position)

        print(f"\n=== Prompt {idx} | pair={item.pair_id} role={item.role} ===")
        print(item.prompt)
        print("\n--- Clean ---")
        print(clean_text)

        for layer in layers:
            for alpha in alphas:
                patched_text = apply_direction(
                    model,
                    item.prompt,
                    direction,
                    layer,
                    args.probe_position,
                    alpha,
                    token_index=token_index,
                    max_new_tokens=args.max_new_tokens,
                )
                print(f"\n--- Patched (layer={layer}, alpha={alpha}) ---")
                print(patched_text)
                outputs.append(
                    {
                        "pair_id": item.pair_id,
                        "role": item.role,
                        "label": item.label,
                        "prompt": item.prompt,
                        "clean_output": clean_text,
                        "patched_output": patched_text,
                        "alpha": alpha,
                        "layer": layer,
                        "probe_position": args.probe_position,
                    }
                )

    Path(args.out_path).write_text("\n".join(json.dumps(o) for o in outputs))
    print("\nWrote:", args.out_path)


if __name__ == "__main__":
    main()
