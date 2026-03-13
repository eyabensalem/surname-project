"""
Simple evaluation of summarization outputs using ROUGE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from rouge_score import rouge_scorer
from config import EVALUATION_RESULTS_FILE, MODEL_COMPARISON_FILE

INPUT_FILE = MODEL_COMPARISON_FILE
OUTPUT_FILE = EVALUATION_RESULTS_FILE



# Manual reference summaries for a few groups
REFERENCE_SUMMARIES = {
    1: "Le nom semble désigner une personne travaillant le métal et peut aussi être lié à une variante de métayer.",
    2: "Le nom signifie en allemand montagne noire et désigne une personne originaire d’une localité portant ce nom.",
    3: "Ce nom est un diminutif lié au prénom Henri ou Jean, et en Bretagne il peut aussi signifier le vieux ou l’aîné.",
}


def load_json(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def evaluate_model(reference: str, generated: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)

    return {
        "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
        "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
    }


def main() -> None:
    data = load_json(INPUT_FILE)
    results: List[Dict[str, Any]] = []

    for item in data:
        group_id = item["group_id"]
        reference = REFERENCE_SUMMARIES.get(group_id)

        if not reference:
            continue

        results.append(
            {
                "group_id": group_id,
                "reference_summary": reference,
                "tfidf_scores": evaluate_model(reference, item.get("tfidf_summary", "")),
                "textrank_scores": evaluate_model(reference, item.get("textrank_summary", "")),
                "bart_scores": evaluate_model(reference, item.get("bart_summary", "")),
            }
        )

    save_json(results, OUTPUT_FILE)
    print(f"Done. File created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()