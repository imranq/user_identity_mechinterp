#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List


VOWELS = "aeiouy"
TECH_TERMS = {
    "algorithm", "analysis", "approximation", "assumption", "causal", "complexity",
    "derivative", "entropy", "equilibrium", "evidence", "gradient", "inference",
    "integral", "logit", "model", "parameter", "probability", "proof", "relativity",
    "theorem", "variance", "vector", "quantum", "dilation", "molecule", "hypothesis",
}


def count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text)


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def compute_metrics(text: str) -> Dict[str, float]:
    tokens = tokenize(text)
    sentences = sentence_split(text)
    if not tokens or not sentences:
        return {
            "words": 0,
            "sentences": 0,
            "avg_sentence_len": 0.0,
            "avg_word_len": 0.0,
            "syllables_per_word": 0.0,
            "flesch_kincaid": 0.0,
            "tech_term_rate": 0.0,
        }
    syllables = sum(count_syllables(t) for t in tokens)
    words = len(tokens)
    sent_count = len(sentences)
    avg_sentence_len = words / sent_count
    avg_word_len = mean(len(t) for t in tokens)
    syllables_per_word = syllables / words
    flesch_kincaid = 0.39 * avg_sentence_len + 11.8 * syllables_per_word - 15.59
    tech_terms = sum(1 for t in tokens if t.lower() in TECH_TERMS)
    tech_term_rate = tech_terms / words
    return {
        "words": words,
        "sentences": sent_count,
        "avg_sentence_len": avg_sentence_len,
        "avg_word_len": avg_word_len,
        "syllables_per_word": syllables_per_word,
        "flesch_kincaid": flesch_kincaid,
        "tech_term_rate": tech_term_rate,
    }


def load_rows(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    agg = defaultdict(list)
    for row in rows:
        clean = row.get("clean_output", "")
        steered = row.get("steered_output", row.get("patched_output", ""))
        clean_m = compute_metrics(clean)
        steered_m = compute_metrics(steered)
        for k, v in clean_m.items():
            agg[f"clean_{k}"].append(v)
        for k, v in steered_m.items():
            agg[f"steered_{k}"].append(v)
        for k in clean_m:
            agg[f"delta_{k}"].append(steered_m[k] - clean_m[k])
    return {k: mean(v) if v else 0.0 for k, v in agg.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple metrics for persona steering outputs.")
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.input_path))
    summary = summarize(rows)
    print("Metrics summary (means across outputs):")
    for key in sorted(summary.keys()):
        print(f"{key}: {summary[key]:.4f}")


if __name__ == "__main__":
    main()
