"""
Merge origin texts according to surname groups.

Person 2 pipeline step:
- Load origins.json
- Load grouped_names.json
- Merge texts corresponding to each group
- Save merged_groups.json
"""

import json
from pathlib import Path


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

ORIGINS_FILE = DATA_DIR / "origins.json"
GROUPED_FILE = RESULTS_DIR / "grouped_names.json"
OUTPUT_FILE = RESULTS_DIR / "merged_groups.json"


def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_texts_by_group(origins, groups):
    """
    Merge origin texts by group.

    Parameters
    ----------
    origins : dict
        Dictionary {origin_id: text}

    groups : list
        List of groups containing origin_ids

    Returns
    -------
    list
        Groups with merged text
    """

    merged_results = []

    for group in groups:

        group_id = group["group_id"]
        variants = group["variants"]
        origin_ids = group["origin_ids"]

        texts = []

        for oid in origin_ids:
            if oid in origins:
                texts.append(origins[oid])

        merged_text = " ".join(texts)

        merged_results.append({
            "group_id": group_id,
            "variants": variants,
            "origin_ids": origin_ids,
            "merged_text": merged_text
        })

    return merged_results


def main():

    print("Loading data...")

    origins = load_json(ORIGINS_FILE)
    groups = load_json(GROUPED_FILE)

    print("Merging texts...")

    merged_groups = merge_texts_by_group(origins, groups)

    print("Saving results...")

    save_json(merged_groups, OUTPUT_FILE)

    print("Done. File created:", OUTPUT_FILE)


if __name__ == "__main__":
    main()