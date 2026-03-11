"""
Generate short summaries for each surname group.

Hybrid approach:
- sentence splitting
- rule-based sentence prioritization
- TF-IDF scoring as a secondary signal
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


RESULTS_DIR = Path("results")

INPUT_FILE = RESULTS_DIR / "merged_groups.json"
OUTPUT_FILE = RESULTS_DIR / "group_summaries.json"


IMPORTANT_KEYWORDS = [
    "signifie",
    "désigne",
    "variante",
    "variantes",
    "forme",
    "formes",
    "dérivé",
    "dérivés",
    "origine",
    "originaire",
    "toponyme",
    "prénom",
    "nom",
]


def load_json(file_path: Path) -> Any:
    """Load JSON content from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, file_path: Path) -> None:
    """Save data to a formatted JSON file."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def split_sentences(text: str) -> List[str]:
    """Split a text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def clean_summary_text(text: str) -> str:
    """
    Clean a generated summary for better readability.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" .", ".").replace(" ,", ",")
    text = text.replace(" :", ":").replace(" ;", ";")

    return text
def compute_keyword_score(sentence: str) -> float:
    """
    Score a sentence according to domain-specific keywords.
    """
    lowered = sentence.lower()
    score = 0.0

    for keyword in IMPORTANT_KEYWORDS:
        if keyword in lowered:
            score += 2.0

    if "signifie" in lowered:
        score += 3.0

    if "variantes" in lowered or "variante" in lowered:
        score += 2.5

    if "désigne" in lowered:
        score += 2.5

    return score


def compute_tfidf_scores(sentences: List[str]) -> np.ndarray:
    """
    Compute TF-IDF sentence scores.
    """
    if not sentences:
        return np.array([])

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(sentences)
    return np.asarray(matrix.sum(axis=1)).ravel()


def rank_sentences(sentences: List[str]) -> List[Tuple[int, float]]:
    """
    Rank sentences using hybrid scoring:
    keyword importance + normalized TF-IDF score.
    """
    if not sentences:
        return []

    tfidf_scores = compute_tfidf_scores(sentences)

    if len(tfidf_scores) > 0 and tfidf_scores.max() > 0:
        normalized_tfidf = tfidf_scores / tfidf_scores.max()
    else:
        normalized_tfidf = tfidf_scores

    ranked = []

    for idx, sentence in enumerate(sentences):
        keyword_score = compute_keyword_score(sentence)
        hybrid_score = keyword_score + float(normalized_tfidf[idx])

        # slight penalty for overly long sentences
        length = max(len(sentence.split()), 1)
        hybrid_score = hybrid_score - (length / 100)

        ranked.append((idx, hybrid_score))

    return ranked


def summarize_text(text: str, top_n: int = 2) -> str:
    """
    Summarize a text by selecting the best sentences.
    """
    sentences = split_sentences(text)

    if not sentences:
        return ""

    if len(sentences) <= top_n:
        return " ".join(sentences)

    ranked = rank_sentences(sentences)
    top_indices = sorted([idx for idx, _ in sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]])

    return " ".join(sentences[idx] for idx in top_indices)


def build_summaries(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build summaries for all groups.
    """
    summaries = []

    for group in groups:
        merged_text = group.get("merged_text", "")
        summary = summarize_text(merged_text, top_n=2)
        summary = clean_summary_text(summary)
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