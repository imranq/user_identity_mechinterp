#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SYSTEM_PROMPT = (
    "You are grading whether outputs become more expert as alpha increases. "
    "Return exactly two lines: first line YES or NO, second line a one-sentence rationale."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare autorater inputs per layer/alpha sweep.")
    parser.add_argument("--input_path", type=str, default="persona_steer_outputs.jsonl")
    parser.add_argument("--output_csv", type=str, default="autorater_layer_alpha.csv")
    parser.add_argument("--max_per_layer", type=int, default=3)
    args = parser.parse_args()

    rows = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["layer"], row["question"])].append(row)

    out_rows = []
    for (layer, question), items in grouped.items():
        items = sorted(items, key=lambda r: float(r["alpha"]))
        clean = items[0]["clean_output"]
        for alpha_row in items:
            if alpha_row["clean_output"] != clean:
                clean = alpha_row["clean_output"]
                break
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CLEAN_OUTPUT:\n{clean}\n\n"
            "STEERED_OUTPUTS (alpha ascending):\n"
        )
        for item in items:
            prompt += f"\n[alpha={item['alpha']}]\n{item['steered_output']}\n"
        out_rows.append(
            {
                "layer": layer,
                "question": question,
                "prompt": prompt,
            }
        )

    if args.max_per_layer > 0:
        limited = []
        per_layer = defaultdict(int)
        for row in out_rows:
            if per_layer[row["layer"]] >= args.max_per_layer:
                continue
            limited.append(row)
            per_layer[row["layer"]] += 1
        out_rows = limited

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "question", "prompt"])
        writer.writeheader()
        writer.writerows(out_rows)

    print("Wrote:", args.output_csv)


if __name__ == "__main__":
    main()
