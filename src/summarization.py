"""
Generate short summaries for each surname group.

Person 2 pipeline step:
- Load merged_groups.json
- Split merged text into sentences
- Score sentences with TF-IDF
- Keep the most informative sentences
- Save group_summaries.json
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


RESULTS_DIR = Path("results")

INPUT_FILE = RESULTS_DIR / "merged_groups.json"
OUTPUT_FILE = RESULTS_DIR / "group_summaries.json"


def load_json(file_path: Path) -> Any:
    """Load JSON content from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, file_path: Path) -> None:
    """Save data to a formatted JSON file."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def split_sentences(text: str) -> List[str]:
    """
    Split a text into sentences.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        List of cleaned sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def summarize_text(text: str, top_n: int = 2) -> str:
    """
    Summarize a text by selecting the most informative sentences using TF-IDF.

    Parameters
    ----------
    text : str
        Input text.
    top_n : int
        Number of sentences to keep.

    Returns
    -------
    str
        Summary text.
    """
    sentences = split_sentences(text)

    if not sentences:
        return ""

    if len(sentences) <= top_n:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(sentences)

    scores = np.asarray(matrix.sum(axis=1)).ravel()

    top_indices = scores.argsort()[-top_n:]
    top_indices = sorted(top_indices)

    summary = " ".join(sentences[index] for index in top_indices)
    return summary


def build_summaries(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build summaries for all merged groups.

    Parameters
    ----------
    groups : List[Dict[str, Any]]
        Input merged groups.

    Returns
    -------
    List[Dict[str, Any]]
        Groups enriched with summaries.
    """
    summaries = []

    for group in groups:
        merged_text = group.get("merged_text", "")
        summary = summarize_text(merged_text, top_n=2)

        summaries.append(
            {
                "group_id": group.get("group_id"),
                "variants": group.get("variants", []),
                "origin_ids": group.get("origin_ids", []),
                "summary": summary,
            }
        )

    return summaries


def main() -> None:
    """Run the summarization pipeline."""
    print("Loading merged groups...")
    merged_groups = load_json(INPUT_FILE)

    print("Generating summaries...")
    summaries = build_summaries(merged_groups)

    print("Saving summaries...")
    save_json(summaries, OUTPUT_FILE)

    print(f"Done. File created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()