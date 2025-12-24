#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

from cot_faithfulness import (
    PUZZLES,
    full_logits_per_layer,
    logit_lens_scores_batch,
)


def run_mode(
    model: HookedTransformer,
    prompts: List[str],
    choices: List[str],
    puzzles: List[dict],
    out_dir: Path,
    tag: str,
    report_preference: bool,
    grid_plot: bool,
    steer_direction: np.ndarray | None,
    steer_layer: int | None,
    steer_alpha: float,
    kl_plot: bool,
    save_curves: bool,
) -> None:
    scores = logit_lens_scores_batch(
        model,
        prompts,
        choices,
        steer_layer=steer_layer,
        steer_alpha=steer_alpha,
        steer_direction=steer_direction,
    )
    baseline_scores = logit_lens_scores_batch(
        model,
        prompts,
        choices,
        steer_layer=None,
        steer_alpha=0.0,
        steer_direction=None,
    )
    base_logits = None
    steer_logits = None
    if kl_plot and steer_direction is not None:
        base_logits = full_logits_per_layer(model, prompts)
        steer_logits = full_logits_per_layer(
            model,
            prompts,
            steer_layer=steer_layer,
            steer_alpha=steer_alpha,
            steer_direction=steer_direction,
        )

    simple_total = 0
    simple_pref = 0
    for idx, (puzzle, sc) in enumerate(zip(puzzles, scores)):
        diffs = np.array(sc[choices[0]]) - np.array(sc[choices[1]])
        base = baseline_scores[idx]
        base_diffs = np.array(base[choices[0]]) - np.array(base[choices[1]])
        if report_preference and "simple_choice" in puzzle:
            simple_total += 1
            preferred = choices[0] if diffs[-1] >= 0 else choices[1]
            if preferred == puzzle["simple_choice"]:
                simple_pref += 1

        if not grid_plot:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(diffs, label="steered diff (A-B)")
            ax.plot(base_diffs, label="baseline diff (A-B)", linestyle="--")
            ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
            ax.scatter([len(diffs) - 1], [diffs[-1]], s=24, color="tab:blue", zorder=3)
            ax.scatter([len(base_diffs) - 1], [base_diffs[-1]], s=24, color="tab:orange", zorder=3)
            ax.set_xlabel("layer")
            ax.set_ylabel("logit diff")
            ax.set_title(f"{puzzle['id']} | {tag}")
            ax.legend()
            fig.tight_layout()
            plot_path = out_dir / f"cot_lens_{puzzle['id']}_{tag}.png"
            fig.savefig(plot_path)
            plt.close(fig)

        if save_curves:
            diff_path = out_dir / f"cot_diff_curve_{puzzle['id']}_{tag}.csv"
            with diff_path.open("w", encoding="utf-8") as f:
                f.write("layer,steered_diff,baseline_diff\n")
                for layer in range(len(diffs)):
                    f.write(f"{layer},{diffs[layer]},{base_diffs[layer]}\n")

        if kl_plot and steer_direction is not None:
            base_logits_i = base_logits[idx]
            steer_logits_i = steer_logits[idx]
            kl_vals = []
            for layer in range(len(base_logits_i)):
                b = torch.tensor(base_logits_i[layer])
                s = torch.tensor(steer_logits_i[layer])
                b_probs = torch.softmax(b, dim=0)
                s_probs = torch.softmax(s, dim=0)
                kl = torch.sum(b_probs * (torch.log(b_probs + 1e-12) - torch.log(s_probs + 1e-12)))
                kl_vals.append(float(kl.item()))
            if save_curves:
                kl_path = out_dir / f"cot_kl_curve_{puzzle['id']}_{tag}.csv"
                with kl_path.open("w", encoding="utf-8") as f:
                    f.write("layer,kl\n")
                    for layer, val in enumerate(kl_vals):
                        f.write(f"{layer},{val}\n")
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(kl_vals, label="KL(base||steered)")
            ax.set_xlabel("layer")
            ax.set_ylabel("KL")
            ax.set_title(f"{puzzle['id']} | {tag} | KL")
            ax.legend()
            fig.tight_layout()
            plot_path = out_dir / f"cot_kl_{puzzle['id']}_{tag}.png"
            fig.savefig(plot_path)
            plt.close(fig)
    if grid_plot and puzzles:
        cols = 2
        rows = int(np.ceil(len(puzzles) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)
        for idx, (puzzle, sc) in enumerate(zip(puzzles, scores)):
            diffs = np.array(sc[choices[0]]) - np.array(sc[choices[1]])
            base = baseline_scores[idx]
            base_diffs = np.array(base[choices[0]]) - np.array(base[choices[1]])
            ax = axes[idx]
            ax.plot(diffs, label="steered diff (A-B)")
            ax.plot(base_diffs, label="baseline diff (A-B)", linestyle="--")
            ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
            ax.scatter([len(diffs) - 1], [diffs[-1]], s=24, color="tab:blue", zorder=3)
            ax.scatter([len(base_diffs) - 1], [base_diffs[-1]], s=24, color="tab:orange", zorder=3)
            ax.set_title(puzzle["id"])
            ax.set_xlabel("layer")
            ax.set_ylabel("logit diff")
        for ax in axes[len(puzzles):]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(tag)
        fig.tight_layout()
        plot_path = out_dir / f"cot_lens_grid_{tag}.png"
        fig.savefig(plot_path)
        plt.close(fig)

    if report_preference and simple_total:
        frac = simple_pref / simple_total
        print(f"[{tag}] Simple-choice preference: {simple_pref}/{simple_total} ({frac:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="CoT faithfulness sweep (single model load).")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=str, default="16,18,20,22")
    parser.add_argument("--alpha", type=float, default=8.0, help="Deprecated, use --alphas.")
    parser.add_argument("--alphas", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="report_artifacts")
    parser.add_argument("--kl_plot", action="store_true")
    parser.add_argument("--save_curves", action="store_true")
    parser.add_argument("--direction_path", type=str, default="report_artifacts/persona_direction.npy")
    parser.add_argument("--tag", type=str, default="", help="Optional tag suffix for filenames.")
    parser.add_argument("--report_preference", action="store_true", help="Report % simple-choice preference.")
    parser.add_argument("--skip_hints", action="store_true", help="Only run no-hint prompts.")
    parser.add_argument("--grid_plot", action="store_true", help="Save a grid plot per config instead of per puzzle.")
    parser.add_argument(
        "--puzzle_ids",
        type=str,
        default="",
        help="Comma-separated puzzle IDs to run (default: all).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if args.alphas:
        alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    else:
        alphas = [args.alpha]
    direction = np.load(args.direction_path)

    puzzles = PUZZLES
    if args.puzzle_ids:
        wanted = {p.strip() for p in args.puzzle_ids.split(",") if p.strip()}
        puzzles = [p for p in PUZZLES if p["id"] in wanted]
    if not puzzles:
        return
    choices = puzzles[0]["choices"]
    for layer in layers:
        for alpha in alphas:
            alpha_tag = str(alpha).replace(".", "p")
            suffix = f"_L{layer}_a{alpha_tag}"
            if args.tag:
                suffix = f"{suffix}_{args.tag}"
            prompts = [f"{puzzle['question']}\nAnswer:" for puzzle in puzzles]
            run_mode(
                model,
                prompts,
                choices,
                puzzles,
                out_dir,
                tag=f"nohint{suffix}",
                report_preference=args.report_preference,
                grid_plot=args.grid_plot,
                steer_direction=direction,
                steer_layer=layer,
                steer_alpha=alpha,
                kl_plot=args.kl_plot,
                save_curves=args.save_curves,
            )
            if not args.skip_hints:
                prompts = [
                    f"{(puzzle['hinted'] or '')}{puzzle['question']}\nAnswer:"
                    for puzzle in puzzles
                ]
                run_mode(
                    model,
                    prompts,
                    choices,
                    puzzles,
                    out_dir,
                    tag=f"incorrect{suffix}",
                    report_preference=args.report_preference,
                    grid_plot=args.grid_plot,
                    steer_direction=direction,
                    steer_layer=layer,
                    steer_alpha=alpha,
                    kl_plot=args.kl_plot,
                    save_curves=args.save_curves,
                )
                correct_hint = "Note: the correct answer is A.\n"
                prompts = [f"{correct_hint}{puzzle['question']}\nAnswer:" for puzzle in puzzles]
                run_mode(
                    model,
                    prompts,
                    choices,
                    puzzles,
                    out_dir,
                    tag=f"correct{suffix}",
                    report_preference=args.report_preference,
                    grid_plot=args.grid_plot,
                    steer_direction=direction,
                    steer_layer=layer,
                    steer_alpha=alpha,
                    kl_plot=args.kl_plot,
                    save_curves=args.save_curves,
                )


if __name__ == "__main__":
    main()
