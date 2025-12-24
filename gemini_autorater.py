#!/usr/bin/env python3
import argparse
import csv
import os
import time
from pathlib import Path

from google import genai
from google.genai import types


SYSTEM_PROMPT = (
    "You are grading whether an answer sounds like an expert or a novice. "
    "Return only one token: EXPERT or NOVICE."
)


def build_prompt(row, prompt_field: str):
    if prompt_field in row:
        return row[prompt_field]
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"PROMPT:\n{row.get('prompt', '')}\n\n"
        f"CLEAN_OUTPUT:\n{row.get('clean_output', '')}\n\n"
        f"PATCHED_OUTPUT:\n{row.get('patched_output', '')}\n\n"
        "Return only one token: EXPERT or NOVICE."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Autorate outputs with Gemini.")
    parser.add_argument("--input_csv", type=str, default="autorater_inputs.csv")
    parser.add_argument("--output_csv", type=str, default="autorater_outputs.csv")
    parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between calls.")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="CSV field to use as full prompt.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "label", "rationale", "layer", "question"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            prompt = build_prompt(row, args.prompt_field)
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
            resp = client.models.generate_content(
                model=args.model,
                contents=contents,
                config=config,
            )
            text = (resp.text or "").strip() if resp else ""
            parts = text.splitlines()
            label = parts[0].strip().split()[0] if parts else ""
            rationale = parts[1].strip() if len(parts) > 1 else ""
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "label": label,
                    "rationale": rationale,
                    "layer": row.get("layer", ""),
                    "question": row.get("question", ""),
                }
            )
            f.flush()
            if args.sleep:
                time.sleep(args.sleep)

    print("Wrote:", output_path)


if __name__ == "__main__":
    main()
