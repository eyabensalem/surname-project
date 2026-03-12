import argparse
import csv
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import spacy
from sentence_transformers import SentenceTransformer
from unidecode import unidecode


TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"
OUTPUTS_DIR = TEST_DIR / "outputs"

APPROACH_DIRECTORIES = {
    "approach_1_name_similarity": "01_name_similarity",
    "approach_2_name_and_context": "02_name_and_context",
    "approach_3_sequence_matcher": "03_sequence_matcher",
    "approach_4_levenshtein": "04_levenshtein",
    "approach_5_soundex": "05_soundex",
    "approach_6_spacy": "06_spacy",
}
APPROACH_FILE_SUFFIXES = {
    "approach_1_name_similarity": "name_similarity",
    "approach_2_name_and_context": "name_and_context",
    "approach_3_sequence_matcher": "sequence_matcher",
    "approach_4_levenshtein": "levenshtein",
    "approach_5_soundex": "soundex",
    "approach_6_spacy": "spacy",
}

MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
SPACY_MODEL = os.getenv("SPACY_MODEL", "fr_core_news_md")
SIMILARITY_THRESHOLD = float(os.getenv("NAME_GROUP_THRESHOLD", "0.78"))
CONTEXT_SIMILARITY_THRESHOLD = float(os.getenv("NAME_CONTEXT_GROUP_THRESHOLD", str(SIMILARITY_THRESHOLD)))
SEQUENCE_MATCHER_THRESHOLD = float(os.getenv("SEQUENCE_MATCHER_THRESHOLD", "0.88"))
LEVENSHTEIN_THRESHOLD = float(os.getenv("LEVENSHTEIN_THRESHOLD", "0.85"))
SOUNDEX_THRESHOLD = float(os.getenv("SOUNDEX_THRESHOLD", "1.0"))
SPACY_SIMILARITY_THRESHOLD = float(os.getenv("SPACY_SIMILARITY_THRESHOLD", "0.75"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_test_data(path: Path) -> dict[str, list[str]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("The test dataset must be a JSON object mapping each name to a list of descriptions.")

    normalized_payload: dict[str, list[str]] = {}
    for name, descriptions in payload.items():
        if not isinstance(name, str):
            raise ValueError("Each test dataset key must be a string.")
        if not isinstance(descriptions, list) or not all(isinstance(item, str) for item in descriptions):
            raise ValueError(f"Each value must be a list of strings. Invalid entry: {name!r}")
        normalized_payload[name] = descriptions
    return normalized_payload


def build_output_filenames(approach_name: str) -> dict[str, str]:
    suffix = APPROACH_FILE_SUFFIXES[approach_name]
    return {
        "structured_csv": f"structured_dataset_{suffix}.csv",
        "structured_json": f"structured_dataset_{suffix}.json",
        "cleaned_json": f"cleaned_dataset_{suffix}.json",
        "groups_json": f"name_groups_{suffix}.json",
        "final_json": f"final_dataset_{suffix}.json",
        "metadata_json": f"run_metadata_{suffix}.json",
    }


def ensure_output_directories(requested_approaches: list[str]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for approach_name in requested_approaches:
        (OUTPUTS_DIR / APPROACH_DIRECTORIES[approach_name]).mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = unidecode(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    lemmas = []
    for token in tokens:
        lemma = token
        if len(token) > 4 and token.endswith("es"):
            lemma = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            lemma = token[:-1]
        elif len(token) > 4 and token.endswith("e"):
            lemma = token[:-1]
        lemmas.append(lemma)
    return lemmas


def preprocess_name(name: str) -> dict:
    normalized = normalize_text(name)
    tokens = tokenize(normalized)
    lemmas = lemmatize_tokens(tokens)
    normalized_name = "".join(lemmas)
    return {
        "original_name": name,
        "normalized_text": normalized,
        "tokens": tokens,
        "lemmas": lemmas,
        "normalized_name": normalized_name,
    }


def build_structured_rows(test_data: dict[str, list[str]]) -> list[dict]:
    rows = []
    description_index = 1

    for name in sorted(test_data):
        descriptions = test_data[name]
        if not descriptions:
            rows.append({"name": name, "origin_id": f"T{description_index:05d}", "origin_text": ""})
            description_index += 1
            continue

        for description in descriptions:
            rows.append(
                {
                    "name": name,
                    "origin_id": f"T{description_index:05d}",
                    "origin_text": description,
                }
            )
            description_index += 1

    return rows


def build_cleaned_rows(structured_rows: list[dict]) -> list[dict]:
    cleaned_rows = []
    for row in structured_rows:
        processed = preprocess_name(row["name"])
        cleaned_rows.append(
            {
                "name": row["name"],
                "origin_id": row["origin_id"],
                "origin_text": row["origin_text"],
                "normalized_text": processed["normalized_text"],
                "tokens": processed["tokens"],
                "lemmas": processed["lemmas"],
                "normalized_name": processed["normalized_name"],
            }
        )
    return cleaned_rows


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


def cosine_similarity(vector_a, vector_b) -> float:
    return sum(a * b for a, b in zip(vector_a, vector_b))


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, start=1):
        current_row = [index_a]
        for index_b, char_b in enumerate(b, start=1):
            insert_cost = current_row[index_b - 1] + 1
            delete_cost = previous_row[index_b] + 1
            replace_cost = previous_row[index_b - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1 - (levenshtein_distance(a, b) / max_len)


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


def load_embedding_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the Hugging Face model. "
            f"Set HF_MODEL_NAME if you want another model, or install/download '{MODEL_NAME}' first."
        ) from exc


def load_spacy_model():
    try:
        return spacy.load(SPACY_MODEL)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the spaCy model. "
            f"Install/download '{SPACY_MODEL}' first or set SPACY_MODEL to an available pipeline."
        ) from exc


def aggregate_variants(cleaned_rows: list[dict]) -> tuple[dict[str, dict], defaultdict[tuple[str, int], list[str]]]:
    variants_by_normalized: dict[str, dict] = {}
    candidate_buckets: defaultdict[tuple[str, int], list[str]] = defaultdict(list)

    for row in cleaned_rows:
        normalized_name = row["normalized_name"]
        if not normalized_name:
            continue

        bucket = variants_by_normalized.setdefault(
            normalized_name,
            {
                "normalized_name": normalized_name,
                "original_names": set(),
                "origin_ids": set(),
                "origin_texts": set(),
            },
        )
        bucket["original_names"].add(row["name"])
        bucket["origin_ids"].add(row["origin_id"])
        if row["origin_text"]:
            bucket["origin_texts"].add(row["origin_text"])

    for normalized_name in sorted(variants_by_normalized):
        candidate_buckets[build_candidate_key(normalized_name)].append(normalized_name)

    return variants_by_normalized, candidate_buckets


def build_name_only_text(bucket: dict) -> str:
    return bucket["normalized_name"]


def build_name_and_context_text(bucket: dict) -> str:
    normalized_contexts = [normalize_text(text) for text in sorted(bucket["origin_texts"])]
    context_text = " ".join(text for text in normalized_contexts if text)
    if not context_text:
        return bucket["normalized_name"]
    return f"name: {bucket['normalized_name']} context: {context_text}"


def build_group_record(
    group_variants: list[str], variants_by_normalized: dict[str, dict], group_index: int, comparison_details: list[dict]
) -> dict:
    group_variants = sorted(set(group_variants))
    all_original_names = sorted(
        {name for variant in group_variants for name in variants_by_normalized[variant]["original_names"]}
    )
    all_origin_ids = sorted(
        {origin_id for variant in group_variants for origin_id in variants_by_normalized[variant]["origin_ids"]}
    )
    all_origin_texts = sorted(
        {text for variant in group_variants for text in variants_by_normalized[variant]["origin_texts"]}
    )
    return {
        "group_id": f"G{group_index:05d}",
        "canonical_name": all_original_names[0] if all_original_names else group_variants[0],
        "normalized_variants": group_variants,
        "variant_names": all_original_names,
        "origin_ids": all_origin_ids,
        "origin_texts": all_origin_texts,
        "comparisons": comparison_details,
    }


def build_normalized_to_group(groups: list[dict]) -> dict[str, dict]:
    normalized_to_group = {}
    for group in groups:
        for variant in group["normalized_variants"]:
            normalized_to_group[variant] = group
    return normalized_to_group


def build_groups_from_embeddings(
    cleaned_rows: list[dict],
    comparison_text_builder,
    similarity_threshold: float,
    approach_label: str,
) -> tuple[list[dict], dict[str, dict], dict]:
    variants_by_normalized, candidate_buckets = aggregate_variants(cleaned_rows)
    normalized_names = sorted(variants_by_normalized)
    comparison_names = [comparison_text_builder(variants_by_normalized[name]) for name in normalized_names]

    model = load_embedding_model()
    embeddings = model.encode(
        comparison_names,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for index, normalized_name in enumerate(normalized_names):
        variants_by_normalized[normalized_name]["embedding"] = embeddings[index]

    def score_builder(source_name: str, _source_bucket: dict, target_name: str, _target_bucket: dict) -> float:
        source_embedding = variants_by_normalized[source_name]["embedding"]
        target_embedding = variants_by_normalized[target_name]["embedding"]
        return cosine_similarity(source_embedding, target_embedding)

    return build_groups_from_scores(
        variants_by_normalized,
        candidate_buckets,
        score_builder=score_builder,
        similarity_threshold=similarity_threshold,
        approach_label=approach_label,
    )


def build_groups_from_scores(
    variants_by_normalized: dict[str, dict],
    candidate_buckets: defaultdict[tuple[str, int], list[str]],
    score_builder,
    similarity_threshold: float,
    approach_label: str,
) -> tuple[list[dict], dict[str, dict], dict]:
    normalized_names = sorted(variants_by_normalized)
    groups = []
    assigned = set()
    group_index = 1
    total_comparisons = 0
    accepted_comparisons = 0

    for normalized_name in normalized_names:
        if normalized_name in assigned:
            continue

        source_bucket = variants_by_normalized[normalized_name]
        key_prefix, key_length = build_candidate_key(normalized_name)
        group_variants = [normalized_name]
        comparison_details = []

        for length in range(key_length - 3, key_length + 4):
            for other_name in candidate_buckets.get((key_prefix, length), []):
                if other_name == normalized_name or other_name in assigned:
                    continue
                if not should_compare(normalized_name, other_name):
                    continue

                total_comparisons += 1
                target_bucket = variants_by_normalized[other_name]
                score = score_builder(normalized_name, source_bucket, other_name, target_bucket)
                if score >= similarity_threshold:
                    accepted_comparisons += 1
                    group_variants.append(other_name)
                    comparison_details.append(
                        {
                            "source": normalized_name,
                            "target": other_name,
                            "score": round(float(score), 4),
                            "decision": "grouped",
                        }
                    )

        for variant in group_variants:
            assigned.add(variant)

        groups.append(build_group_record(group_variants, variants_by_normalized, group_index, comparison_details))
        group_index += 1

    diagnostics = {
        "approach": approach_label,
        "similarity_threshold": similarity_threshold,
        "total_comparisons": total_comparisons,
        "accepted_comparisons": accepted_comparisons,
        "rejected_comparisons": total_comparisons - accepted_comparisons,
    }
    return groups, build_normalized_to_group(groups), diagnostics


def build_groups_with_sequence_matcher(cleaned_rows: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    variants_by_normalized, candidate_buckets = aggregate_variants(cleaned_rows)

    def score_builder(source_name: str, _source_bucket: dict, target_name: str, _target_bucket: dict) -> float:
        return SequenceMatcher(None, source_name, target_name).ratio()

    return build_groups_from_scores(
        variants_by_normalized,
        candidate_buckets,
        score_builder=score_builder,
        similarity_threshold=SEQUENCE_MATCHER_THRESHOLD,
        approach_label="approach_3_sequence_matcher",
    )


def build_groups_with_levenshtein(cleaned_rows: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    variants_by_normalized, candidate_buckets = aggregate_variants(cleaned_rows)

    def score_builder(source_name: str, _source_bucket: dict, target_name: str, _target_bucket: dict) -> float:
        return levenshtein_ratio(source_name, target_name)

    return build_groups_from_scores(
        variants_by_normalized,
        candidate_buckets,
        score_builder=score_builder,
        similarity_threshold=LEVENSHTEIN_THRESHOLD,
        approach_label="approach_4_levenshtein",
    )


def build_groups_with_soundex(cleaned_rows: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    variants_by_normalized, candidate_buckets = aggregate_variants(cleaned_rows)
    for normalized_name, bucket in variants_by_normalized.items():
        bucket["soundex"] = soundex(normalized_name)

    def score_builder(_source_name: str, source_bucket: dict, _target_name: str, target_bucket: dict) -> float:
        if source_bucket["soundex"] and source_bucket["soundex"] == target_bucket["soundex"]:
            return 1.0
        return 0.0

    return build_groups_from_scores(
        variants_by_normalized,
        candidate_buckets,
        score_builder=score_builder,
        similarity_threshold=SOUNDEX_THRESHOLD,
        approach_label="approach_5_soundex",
    )


def build_groups_with_spacy(cleaned_rows: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    variants_by_normalized, candidate_buckets = aggregate_variants(cleaned_rows)
    nlp = load_spacy_model()
    normalized_names = sorted(variants_by_normalized)
    comparison_texts = [build_name_and_context_text(variants_by_normalized[name]) for name in normalized_names]

    for normalized_name, doc in zip(normalized_names, nlp.pipe(comparison_texts), strict=True):
        variants_by_normalized[normalized_name]["spacy_doc"] = doc

    def score_builder(source_name: str, _source_bucket: dict, target_name: str, _target_bucket: dict) -> float:
        source_doc = variants_by_normalized[source_name]["spacy_doc"]
        target_doc = variants_by_normalized[target_name]["spacy_doc"]
        return source_doc.similarity(target_doc)

    return build_groups_from_scores(
        variants_by_normalized,
        candidate_buckets,
        score_builder=score_builder,
        similarity_threshold=SPACY_SIMILARITY_THRESHOLD,
        approach_label="approach_6_spacy",
    )


def build_final_dataset(cleaned_rows: list[dict], normalized_to_group: dict[str, dict]) -> list[dict]:
    final_by_group: dict[str, dict] = {}

    for row in cleaned_rows:
        group = normalized_to_group.get(row["normalized_name"])
        if group is None:
            continue

        raw_group_id = group["group_id"]
        numeric_group_id = int(raw_group_id[1:]) if raw_group_id.startswith("G") else raw_group_id
        item = final_by_group.setdefault(
            raw_group_id,
            {
                "group_id": numeric_group_id,
                "variants": set(group["variant_names"]),
                "origin_ids": set(),
                "origin_texts": set(),
            },
        )

        item["origin_ids"].add(row["origin_id"])
        if row["origin_text"]:
            item["origin_texts"].add(row["origin_text"])

    final_dataset = []
    for group_id in sorted(final_by_group):
        item = final_by_group[group_id]
        final_dataset.append(
            {
                "group_id": item["group_id"],
                "variants": sorted(item["variants"]),
                "origin_ids": sorted(item["origin_ids"]),
                "origin_texts": sorted(item["origin_texts"]),
            }
        )
    return final_dataset


def run_approach(approach_name: str, cleaned_rows: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    if approach_name == "approach_1_name_similarity":
        return build_groups_from_embeddings(
            cleaned_rows,
            comparison_text_builder=build_name_only_text,
            similarity_threshold=SIMILARITY_THRESHOLD,
            approach_label=approach_name,
        )

    if approach_name == "approach_2_name_and_context":
        return build_groups_from_embeddings(
            cleaned_rows,
            comparison_text_builder=build_name_and_context_text,
            similarity_threshold=CONTEXT_SIMILARITY_THRESHOLD,
            approach_label=approach_name,
        )

    if approach_name == "approach_3_sequence_matcher":
        return build_groups_with_sequence_matcher(cleaned_rows)

    if approach_name == "approach_4_levenshtein":
        return build_groups_with_levenshtein(cleaned_rows)

    if approach_name == "approach_5_soundex":
        return build_groups_with_soundex(cleaned_rows)

    if approach_name == "approach_6_spacy":
        return build_groups_with_spacy(cleaned_rows)

    raise ValueError(f"Unsupported approach '{approach_name}'.")


def write_outputs(
    approach_name: str,
    input_path: Path,
    structured_rows: list[dict],
    cleaned_rows: list[dict],
    groups: list[dict],
    final_dataset: list[dict],
    diagnostics: dict,
) -> Path:
    output_files = build_output_filenames(approach_name)
    output_dir = OUTPUTS_DIR / APPROACH_DIRECTORIES[approach_name]

    write_csv(
        output_dir / output_files["structured_csv"],
        structured_rows,
        ["name", "origin_id", "origin_text"],
    )
    write_json(output_dir / output_files["structured_json"], structured_rows)
    write_json(output_dir / output_files["cleaned_json"], cleaned_rows)
    write_json(output_dir / output_files["groups_json"], groups)
    write_json(output_dir / output_files["final_json"], final_dataset)
    write_json(
        output_dir / output_files["metadata_json"],
        {
            "approach_name": approach_name,
            "input_mode": "test_data_json",
            "input_file": str(input_path),
            "model_name": MODEL_NAME,
            "spacy_model": SPACY_MODEL,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "context_similarity_threshold": CONTEXT_SIMILARITY_THRESHOLD,
            "sequence_matcher_threshold": SEQUENCE_MATCHER_THRESHOLD,
            "levenshtein_threshold": LEVENSHTEIN_THRESHOLD,
            "soundex_threshold": SOUNDEX_THRESHOLD,
            "spacy_similarity_threshold": SPACY_SIMILARITY_THRESHOLD,
            "batch_size": BATCH_SIZE,
            "row_count": len(structured_rows),
            "unique_name_count": len({row["name"] for row in structured_rows}),
            "group_count": len(groups),
            "final_group_count": len(final_dataset),
            "diagnostics": diagnostics,
            "output_dir": str(output_dir),
            "output_files": output_files,
        },
    )
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the six grouping approaches on test/data/test_data.json and write outputs under test/outputs."
    )
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "test_data.json"),
        help="Path to the test dataset JSON file.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=list(APPROACH_DIRECTORIES),
        help="Approach names to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_approaches = args.approaches
    invalid = [name for name in requested_approaches if name not in APPROACH_DIRECTORIES]
    if invalid:
        valid_values = ", ".join(sorted(APPROACH_DIRECTORIES))
        raise ValueError(f"Unknown approach(es): {', '.join(invalid)}. Expected values: {valid_values}")

    input_path = Path(args.input)
    test_data = load_test_data(input_path)
    structured_rows = build_structured_rows(test_data)
    cleaned_rows = build_cleaned_rows(structured_rows)

    ensure_output_directories(requested_approaches)

    summary = []
    for approach_name in requested_approaches:
        groups, normalized_to_group, diagnostics = run_approach(approach_name, cleaned_rows)
        final_dataset = build_final_dataset(cleaned_rows, normalized_to_group)
        output_dir = write_outputs(
            approach_name,
            input_path,
            structured_rows,
            cleaned_rows,
            groups,
            final_dataset,
            diagnostics,
        )
        summary.append(
            {
                "approach_name": approach_name,
                "output_dir": str(output_dir),
                "group_count": len(groups),
                "final_group_count": len(final_dataset),
            }
        )
        print(f"Created outputs for {approach_name} in {output_dir}")

    write_json(OUTPUTS_DIR / "run_summary.json", summary)
    print(f"Created {OUTPUTS_DIR / 'run_summary.json'}")


if __name__ == "__main__":
    main()
