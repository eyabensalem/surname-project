import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_default_prediction_path(preferred_paths: list[Path]) -> Path:
    for path in preferred_paths:
        if path.exists():
            return path
    return preferred_paths[0]


def extract_variant_names(group: dict) -> list[str]:
    if "variants" in group:
        return group["variants"]
    if "variant_names" in group:
        return group["variant_names"]
    raise ValueError("Each group must contain either 'variants' or 'variant_names'.")


def build_gold_mapping(gold_groups: list[dict]) -> dict[str, str]:
    gold_mapping = {}
    for group in gold_groups:
        gold_group_id = group["gold_group_id"]
        for name in extract_variant_names(group):
            if name in gold_mapping:
                raise ValueError(f"Duplicate gold annotation for '{name}'.")
            gold_mapping[name] = gold_group_id
    return gold_mapping


def build_predicted_mapping(predicted_groups: list[dict]) -> dict[str, str]:
    predicted_mapping = {}
    for index, group in enumerate(predicted_groups, start=1):
        group_id = str(group.get("group_id", index))
        for name in extract_variant_names(group):
            predicted_mapping[name] = group_id
    return predicted_mapping


def build_contingency(names: list[str], gold_mapping: dict[str, str], predicted_mapping: dict[str, str]):
    contingency: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    predicted_sizes: defaultdict[str, int] = defaultdict(int)
    gold_sizes: defaultdict[str, int] = defaultdict(int)

    for name in names:
        gold_label = gold_mapping[name]
        predicted_label = predicted_mapping.get(name, f"__missing__:{name}")
        contingency[predicted_label][gold_label] += 1
        predicted_sizes[predicted_label] += 1
        gold_sizes[gold_label] += 1

    return contingency, predicted_sizes, gold_sizes


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_pairwise_metrics(gold_mapping: dict[str, str], predicted_mapping: dict[str, str]) -> dict[str, float]:
    names = sorted(gold_mapping)
    contingency, predicted_sizes, gold_sizes = build_contingency(names, gold_mapping, predicted_mapping)

    true_positive = sum(comb(count, 2) for row in contingency.values() for count in row.values() if count > 1)
    predicted_positive = sum(comb(size, 2) for size in predicted_sizes.values() if size > 1)
    gold_positive = sum(comb(size, 2) for size in gold_sizes.values() if size > 1)

    false_positive = predicted_positive - true_positive
    false_negative = gold_positive - true_positive

    precision = safe_divide(true_positive, predicted_positive)
    recall = safe_divide(true_positive, gold_positive)
    f1 = safe_divide(2 * precision * recall, precision + recall) if precision or recall else 0.0

    false_merge_rate = safe_divide(false_positive, predicted_positive)
    false_split_rate = safe_divide(false_negative, gold_positive)

    return {
        "evaluated_name_count": len(names),
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "false_negative_pairs": false_negative,
        "predicted_positive_pairs": predicted_positive,
        "gold_positive_pairs": gold_positive,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "false_merge_rate": round(false_merge_rate, 6),
        "false_split_rate": round(false_split_rate, 6),
    }


def format_metrics(label: str, metrics: dict[str, float]) -> str:
    return "\n".join(
        [
            label,
            f"  evaluated_name_count : {metrics['evaluated_name_count']}",
            f"  precision            : {metrics['precision']}",
            f"  recall               : {metrics['recall']}",
            f"  f1_score             : {metrics['f1_score']}",
            f"  false_merge_rate     : {metrics['false_merge_rate']}",
            f"  false_split_rate     : {metrics['false_split_rate']}",
            f"  true_positive_pairs  : {metrics['true_positive_pairs']}",
            f"  false_positive_pairs : {metrics['false_positive_pairs']}",
            f"  false_negative_pairs : {metrics['false_negative_pairs']}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pairwise clustering metrics for the two grouping approaches.")
    parser.add_argument(
        "--gold",
        default=str(DATA_DIR / "gold_clusters.json"),
        help="Path to the manually annotated gold clusters JSON file.",
    )
    parser.add_argument(
        "--approach1",
        default=str(
            resolve_default_prediction_path(
                [
                    OUTPUTS_DIR / "01_name_similarity" / "final_dataset_name_similarity.json",
                    OUTPUTS_DIR / "01_name_similarity" / "final_dataset1.json",
                ]
            )
        ),
        help="Path to the final dataset for approach 1.",
    )
    parser.add_argument(
        "--approach2",
        default=str(
            resolve_default_prediction_path(
                [
                    OUTPUTS_DIR / "02_name_and_context" / "final_dataset_name_and_context.json",
                    OUTPUTS_DIR / "02_name_and_context" / "final_dataset2.json",
                ]
            )
        ),
        help="Path to the final dataset for approach 2.",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold file not found: {gold_path}. Create it from data/gold_clusters.template.json first."
        )

    approach_1_path = Path(args.approach1)
    approach_2_path = Path(args.approach2)
    if not approach_1_path.exists():
        raise FileNotFoundError(f"Approach 1 file not found: {approach_1_path}")
    if not approach_2_path.exists():
        raise FileNotFoundError(f"Approach 2 file not found: {approach_2_path}")

    gold_mapping = build_gold_mapping(load_json(gold_path))
    approach_1_metrics = compute_pairwise_metrics(gold_mapping, build_predicted_mapping(load_json(approach_1_path)))
    approach_2_metrics = compute_pairwise_metrics(gold_mapping, build_predicted_mapping(load_json(approach_2_path)))

    print(format_metrics("Approach 1: name similarity", approach_1_metrics))
    print()
    print(format_metrics("Approach 2: name + context", approach_2_metrics))


if __name__ == "__main__":
    main()
