"""
Group first names with Soundex and generate group summaries.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict

from config import (
    FIRSTNAMES_DATASET_FILE,
    FIRSTNAMES_GROUP_SUMMARIES_FILE,
    FIRSTNAMES_GROUPED_FILE,
)

SUMMARY_TOP_N = 2
IMPORTANT_SUMMARY_KEYWORDS = [
    "origine",
    "signifie",
    "signification",
    "variante",
    "derive",
    "prenom",
    "biblique",
    "hebreu",
    "latin",
    "arabe",
    "grec",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_summary_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    return text


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def soundex(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    letters = normalized.replace(" ", "")
    if not letters:
        return ""

    first_letter = letters[0].upper()
    mapping = {
        "b": "1",
        "f": "1",
        "p": "1",
        "v": "1",
        "c": "2",
        "g": "2",
        "j": "2",
        "k": "2",
        "q": "2",
        "s": "2",
        "x": "2",
        "z": "2",
        "d": "3",
        "t": "3",
        "l": "4",
        "m": "5",
        "n": "5",
        "r": "6",
    }

    encoded = [first_letter]
    previous_digit = mapping.get(letters[0], "")
    for char in letters[1:]:
        digit = mapping.get(char, "")
        if digit and digit != previous_digit:
            encoded.append(digit)
        previous_digit = digit

    code = "".join(encoded)[:4]
    return code.ljust(4, "0")


def build_candidate_key(name: str) -> tuple[str, int]:
    prefix = name[:3] if len(name) >= 3 else name
    return prefix, len(name)


def should_compare(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a[0] != b[0]:
        return False
    if abs(len(a) - len(b)) > 3:
        return False
    return True


def unique_non_empty(values) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        cleaned = clean_summary_text(str(value)) if value else ""
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(cleaned)
    return ordered


def build_merged_text(group: dict) -> str:
    parts = []

    if group["origins"]:
        parts.append("Origines possibles: " + ", ".join(group["origins"]) + ".")

    if group["meanings"]:
        parts.append("Significations relevees: " + " ".join(group["meanings"][:3]))

    for description in group["descriptions"]:
        if description not in parts:
            parts.append(description)

    return clean_summary_text(" ".join(parts))


def rank_sentences(sentences: list[str]) -> list[tuple[int, float]]:
    ranked = []
    for index, sentence in enumerate(sentences):
        lowered = normalize_text(sentence)
        keyword_score = sum(1 for keyword in IMPORTANT_SUMMARY_KEYWORDS if keyword in lowered)
        length_score = min(len(sentence) / 120, 1.0)
        position_bonus = 1 / (index + 1)
        ranked.append((index, keyword_score + length_score + position_bonus))
    return ranked


def summarize_text(text: str, top_n: int = SUMMARY_TOP_N) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

    ranked = rank_sentences(sentences)
    selected_indices = sorted(
        index for index, _score in sorted(ranked, key=lambda item: item[1], reverse=True)[:top_n]
    )
    return clean_summary_text(" ".join(sentences[index] for index in selected_indices))


def average_quality_score(entries: list[dict]) -> float:
    scores = [entry.get("quality_score", 0) for entry in entries if isinstance(entry.get("quality_score", 0), (int, float))]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


def build_groups(entries: list[dict]) -> list[dict]:
    entries_by_normalized = {}
    candidate_buckets = defaultdict(list)

    for entry in entries:
        first_name = clean_summary_text(entry.get("first_name", ""))
        if not first_name:
            continue

        normalized_name = normalize_text(first_name)
        if not normalized_name:
            continue

        bucket = entries_by_normalized.setdefault(
            normalized_name,
            {
                "soundex_code": soundex(first_name),
                "entries": [],
                "variants": [],
            },
        )
        bucket["entries"].append(entry)
        bucket["variants"].append(first_name)

    for normalized_name in entries_by_normalized:
        candidate_buckets[build_candidate_key(normalized_name)].append(normalized_name)

    groups = []
    assigned = set()

    group_index = 1
    for normalized_name in sorted(entries_by_normalized):
        if normalized_name in assigned:
            continue

        source_bucket = entries_by_normalized[normalized_name]
        key_prefix, key_length = build_candidate_key(normalized_name)
        group_names = [normalized_name]

        for length in range(key_length - 3, key_length + 4):
            for other_name in candidate_buckets.get((key_prefix, length), []):
                if other_name == normalized_name or other_name in assigned:
                    continue
                if not should_compare(normalized_name, other_name):
                    continue

                target_bucket = entries_by_normalized[other_name]
                if source_bucket["soundex_code"] and source_bucket["soundex_code"] == target_bucket["soundex_code"]:
                    group_names.append(other_name)

        group_entries = []
        group_variants = []
        for grouped_name in group_names:
            assigned.add(grouped_name)
            group_entries.extend(entries_by_normalized[grouped_name]["entries"])
            group_variants.extend(entries_by_normalized[grouped_name]["variants"])

        variants = unique_non_empty(group_variants)
        origins = unique_non_empty(entry.get("origin", "") for entry in group_entries)
        meanings = unique_non_empty(entry.get("meaning", "") for entry in group_entries)
        descriptions = unique_non_empty(entry.get("description", "") for entry in group_entries)
        source_urls = unique_non_empty(entry.get("url", "") for entry in group_entries)

        group = {
            "group_id": f"firstname_soundex_{group_index:04d}",
            "display_name": variants[0] if variants else "",
            "soundex_code": source_bucket["soundex_code"],
            "variants": variants,
            "origins": origins,
            "meanings": meanings,
            "entry_count": len(group_entries),
            "source_urls": source_urls,
            "descriptions": descriptions,
            "quality_score": average_quality_score(group_entries),
        }
        group["merged_text"] = build_merged_text(group)
        groups.append(group)
        group_index += 1

    return groups


def build_group_summaries(groups: list[dict]) -> list[dict]:
    summaries = []
    for group in groups:
        summaries.append(
            {
                "group_id": group["group_id"],
                "display_name": group["display_name"],
                "soundex_code": group["soundex_code"],
                "variants": group["variants"],
                "origins": group["origins"],
                "meanings": group["meanings"],
                "entry_count": group["entry_count"],
                "summary": summarize_text(group.get("merged_text", "")),
                "quality_score": group["quality_score"],
            }
        )
    return summaries


def main():
    print("Loading firstname dataset...")
    dataset = load_json(FIRSTNAMES_DATASET_FILE)
    print(f"{len(dataset)} entries loaded")

    grouped_firstnames = build_groups(dataset)
    group_summaries = build_group_summaries(grouped_firstnames)

    write_json(FIRSTNAMES_GROUPED_FILE, grouped_firstnames)
    print(f"Saved grouped firstnames: {FIRSTNAMES_GROUPED_FILE}")

    write_json(FIRSTNAMES_GROUP_SUMMARIES_FILE, group_summaries)
    print(f"Saved firstname group summaries: {FIRSTNAMES_GROUP_SUMMARIES_FILE}")


if __name__ == "__main__":
    main()
