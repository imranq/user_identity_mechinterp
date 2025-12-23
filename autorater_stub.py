import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict


SYSTEM_PROMPT = (
    "You are grading whether an answer sounds like an expert or a novice. "
    "Return only one token: EXPERT or NOVICE."
)


def load_outputs(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare outputs for autorating")
    parser.add_argument("--input_path", type=str, default="patched_outputs.jsonl")
    parser.add_argument("--output_csv", type=str, default="autorater_inputs.csv")
    args = parser.parse_args()

    rows = load_outputs(args.input_path)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "system_prompt", "prompt", "clean_output", "patched_output"]
        )
        for idx, row in enumerate(rows):
            writer.writerow(
                [
                    idx,
                    SYSTEM_PROMPT,
                    row["prompt"],
                    row["clean_output"],
                    row["patched_output"],
                ]
            )

    print("Wrote autorater inputs to", args.output_csv)


if __name__ == "__main__":
    main()
