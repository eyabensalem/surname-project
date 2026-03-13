"""
Generate summaries for first-name descriptions.

This script reads the scraped first-name dataset and generates
short summaries using a simple NLP extractive method.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict


INPUT_FILE = Path("results/firstnames_dataset.json")
OUTPUT_FILE = Path("results/firstnames_summaries.json")
from config import FIRSTNAMES_DATASET_FILE, FIRSTNAMES_SUMMARIES_FILE

INPUT_FILE = FIRSTNAMES_DATASET_FILE
OUTPUT_FILE = FIRSTNAMES_SUMMARIES_FILE

def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def summarize_text(text: str, max_sentences: int = 2) -> str:
    """
    Simple extractive summarization.

    We take the first informative sentences.
    """
    sentences = split_sentences(text)

    if not sentences:
        return ""

    summary_sentences = sentences[:max_sentences]

    return ". ".join(summary_sentences) + "."


def process_firstnames(data: List[Dict]) -> List[Dict]:
    """Generate summaries for all first names."""
    results = []

    for item in data:

        description = item.get("description", "")

        if not description:
            continue

        summary = summarize_text(description)

        results.append(
            {
                "first_name": item.get("first_name"),
                "origin": item.get("origin", ""),
                "meaning": item.get("meaning", ""),
                "summary": summary,
                "quality_score": item.get("quality_score", 0),
            }
        )

    return results


def main():

    print("Loading firstname dataset...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"{len(data)} entries loaded")

    summaries = process_firstnames(data)

    print(f"{len(summaries)} summaries generated")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"Saved file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()