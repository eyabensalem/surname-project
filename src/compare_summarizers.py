"""
Compare several summarization approaches:
- TF-IDF + keyword scoring
- TextRank
- BART (transformer)

Input:
    results/merged_groups.json

Output:
    results/model_comparison.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline


RESULTS_DIR = Path("results")
INPUT_FILE = RESULTS_DIR / "merged_groups.json"
OUTPUT_FILE = RESULTS_DIR / "model_comparison.json"

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
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    return text


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def compute_keyword_score(sentence: str) -> float:
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


def tfidf_summary(text: str, top_n: int = 2) -> str:
    sentences = split_sentences(text)

    if not sentences:
        return ""
    if len(sentences) <= top_n:
        return clean_text(" ".join(sentences))

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(sentences)
    tfidf_scores = np.asarray(matrix.sum(axis=1)).ravel()

    if len(tfidf_scores) > 0 and tfidf_scores.max() > 0:
        normalized_tfidf = tfidf_scores / tfidf_scores.max()
    else:
        normalized_tfidf = tfidf_scores

    ranked = []
    for idx, sentence in enumerate(sentences):
        keyword_score = compute_keyword_score(sentence)
        score = keyword_score + float(normalized_tfidf[idx])
        score -= len(sentence.split()) / 100
        ranked.append((idx, score))

    top_indices = sorted(
        [idx for idx, _ in sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]]
    )

    return clean_text(" ".join(sentences[idx] for idx in top_indices))


def textrank_summary(text: str, sentence_count: int = 2) -> str:
    if not text.strip():
        return ""

    parser = PlaintextParser.from_string(text, Tokenizer("french"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)

    summary = " ".join(str(sentence) for sentence in summary_sentences)
    return clean_text(summary)


def chunk_text(text: str, max_words: int = 180) -> List[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def bart_summary(text: str, summarizer_pipeline) -> str:
    if not text.strip():
        return ""

    chunks = chunk_text(text, max_words=180)
    partial_summaries = []

    for chunk in chunks[:3]:
        try:
            result = summarizer_pipeline(
                chunk,
                max_length=60,
                min_length=15,
                do_sample=False
            )
            partial_summaries.append(result[0]["summary_text"])
        except Exception:
            partial_summaries.append(chunk)

    merged = " ".join(partial_summaries)
    return clean_text(merged)


def compare_models(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summarizer_pipeline = pipeline(model="sshleifer/distilbart-cnn-12-6")

    results = []

    for group in groups:
        text = group.get("merged_text", "")

        item = {
            "group_id": group.get("group_id"),
            "variants": group.get("variants", []),
            "original_text": text,
            "tfidf_summary": tfidf_summary(text, top_n=2),
            "textrank_summary": textrank_summary(text, sentence_count=2),
            "bart_summary": bart_summary(text, summarizer_pipeline),
        }

        results.append(item)

    return results


def main() -> None:
    print("Loading merged groups...")
    groups = load_json(INPUT_FILE)

    print("Comparing summarization models...")
    results = compare_models(groups)

    print("Saving comparison file...")
    save_json(results, OUTPUT_FILE)

    print(f"Done. File created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()