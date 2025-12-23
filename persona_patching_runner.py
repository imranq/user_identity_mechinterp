import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from persona_prompts import build_prompt_dataset
from probe_user_representation import get_probe_token_index


def apply_direction(
    model: HookedTransformer,
    prompt: str,
    direction: np.ndarray,
    layer: int,
    probe_position: str,
    alpha: float,
) -> str:
    direction_t = torch.tensor(direction, device=model.cfg.device, dtype=torch.float32)

    def patch_hook(activation, hook):
        token_index = get_probe_token_index(model, prompt, probe_position)
        activation[0, token_index] = activation[0, token_index] + alpha * direction_t
        return activation

    tokens = model.to_tokens(prompt)
    hook_name = f"blocks.{layer}.hook_resid_pre"
    try:
        with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
            patched_tokens = model.generate(
                tokens,
                max_new_tokens=80,
                do_sample=False,
            )
    except TypeError:
        patched_tokens = model.generate(
            tokens,
            max_new_tokens=80,
            do_sample=False,
            hooks=[(hook_name, patch_hook)],
        )
    return model.to_string(patched_tokens[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run persona patching with a direction")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--direction_path", type=str, default="persona_direction.npy")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--probe_position", type=str, default="question_last_token")
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--n_questions_per_pair", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--align_probe_index", action="store_true")
    parser.add_argument("--probe_template_id", type=int, default=0)
    parser.add_argument("--persona_pad_token", type=str, default=" X")
    parser.add_argument("--out_path", type=str, default="patched_outputs.jsonl")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("Model:", args.model_name)
    print("Device:", device)
    print("Layer:", args.layer)
    print("Probe position:", args.probe_position)
    print("Alpha:", args.alpha)

    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    direction = np.load(args.direction_path)
    print("Direction shape:", direction.shape)

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

    outputs: List[Dict[str, str]] = []
    for item in tqdm(prompt_objs, desc="Patching prompts", unit="prompt"):
        prompt = item.prompt
        tokens = model.to_tokens(prompt)
        clean_tokens = model.generate(tokens, max_new_tokens=80, do_sample=False)
        clean_text = model.to_string(clean_tokens[0])
        patched_text = apply_direction(
            model,
            prompt,
            direction,
            args.layer,
            args.probe_position,
            args.alpha,
        )
        outputs.append(
            {
                "pair_id": item.pair_id,
                "role": item.role,
                "label": item.label,
                "prompt": prompt,
                "clean_output": clean_text,
                "patched_output": patched_text,
                "alpha": args.alpha,
                "layer": args.layer,
                "probe_position": args.probe_position,
            }
        )

    Path(args.out_path).write_text("\n".join(json.dumps(o) for o in outputs))
    print("Total outputs:", len(outputs))
    print("Saved outputs to", args.out_path)


if __name__ == "__main__":
    main()
