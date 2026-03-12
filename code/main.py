import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers import SentenceTransformer
from unidecode import unidecode


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUTS_DIR = BASE_DIR.parent / "outputs"
APPROACH_DIRECTORIES = {
    "approach_1_name_similarity": "01_name_similarity",
    "approach_2_name_and_context": "02_name_and_context",
}
APPROACH_FILE_SUFFIXES = {
    "approach_1_name_similarity": "name_similarity",
    "approach_2_name_and_context": "name_and_context",
}
APPROACH_NAME = os.getenv("NAME_GROUP_APPROACH", "approach_1_name_similarity")
MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = float(os.getenv("NAME_GROUP_THRESHOLD", "0.78"))
CONTEXT_SIMILARITY_THRESHOLD = float(os.getenv("NAME_CONTEXT_GROUP_THRESHOLD", str(SIMILARITY_THRESHOLD)))
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


def ensure_output_directory() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for folder_name in APPROACH_DIRECTORIES.values():
        (OUTPUTS_DIR / folder_name).mkdir(parents=True, exist_ok=True)

    try:
        folder_name = APPROACH_DIRECTORIES[APPROACH_NAME]
    except KeyError as exc:
        valid_values = ", ".join(sorted(APPROACH_DIRECTORIES))
        raise ValueError(f"Unknown NAME_GROUP_APPROACH '{APPROACH_NAME}'. Expected one of: {valid_values}") from exc

    return OUTPUTS_DIR / folder_name


def build_output_filenames() -> dict[str, str]:
    try:
        suffix = APPROACH_FILE_SUFFIXES[APPROACH_NAME]
    except KeyError as exc:
        valid_values = ", ".join(sorted(APPROACH_FILE_SUFFIXES))
        raise ValueError(f"Unknown NAME_GROUP_APPROACH '{APPROACH_NAME}'. Expected one of: {valid_values}") from exc

    return {
        "structured_csv": f"structured_dataset_{suffix}.csv",
        "structured_json": f"structured_dataset_{suffix}.json",
        "cleaned_json": f"cleaned_dataset_{suffix}.json",
        "groups_json": f"name_groups_{suffix}.json",
        "final_json": f"final_dataset_{suffix}.json",
        "metadata_json": f"run_metadata_{suffix}.json",
    }


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


def build_structured_rows(names: list[dict], origins: dict[str, str]) -> list[dict]:
    rows = []
    for entry in names:
        for origin_id in entry["origins"]:
            rows.append(
                {
                    "name": entry["name"],
                    "origin_id": origin_id,
                    "origin_text": origins.get(origin_id, ""),
                }
            )
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


def load_embedding_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the Hugging Face model. "
            f"Set HF_MODEL_NAME if you want another model, or install/download '{MODEL_NAME}' first."
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


def build_groups(
    cleaned_rows: list[dict],
    comparison_text_builder,
    similarity_threshold: float,
    approach_label: str,
) -> tuple[list[dict], dict[str, dict], dict]:
    variants_by_normalized, candidate_buckets = aggregate_variants(cleaned_rows)
    normalized_names = sorted(variants_by_normalized)
    comparison_names = []

    for normalized_name in normalized_names:
        comparison_names.append(comparison_text_builder(variants_by_normalized[normalized_name]))

    model = load_embedding_model()
    embeddings = model.encode(
        comparison_names,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for index, normalized_name in enumerate(normalized_names):
        variants_by_normalized[normalized_name]["embedding"] = embeddings[index]

    groups = []
    assigned = set()
    group_index = 1
    total_comparisons = 0
    accepted_comparisons = 0

    for normalized_name in normalized_names:
        if normalized_name in assigned:
            continue

        key_prefix, key_length = build_candidate_key(normalized_name)
        group_variants = [normalized_name]
        comparison_details = []
        source_embedding = variants_by_normalized[normalized_name]["embedding"]

        for length in range(key_length - 3, key_length + 4):
            for other_name in candidate_buckets.get((key_prefix, length), []):
                if other_name == normalized_name or other_name in assigned:
                    continue
                if not should_compare(normalized_name, other_name):
                    continue

                total_comparisons += 1
                target_embedding = variants_by_normalized[other_name]["embedding"]
                score = cosine_similarity(source_embedding, target_embedding)
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


def run_selected_approach(cleaned_rows: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    if APPROACH_NAME == "approach_1_name_similarity":
        return build_groups(
            cleaned_rows,
            comparison_text_builder=build_name_only_text,
            similarity_threshold=SIMILARITY_THRESHOLD,
            approach_label=APPROACH_NAME,
        )

    if APPROACH_NAME == "approach_2_name_and_context":
        return build_groups(
            cleaned_rows,
            comparison_text_builder=build_name_and_context_text,
            similarity_threshold=CONTEXT_SIMILARITY_THRESHOLD,
            approach_label=APPROACH_NAME,
        )

    raise ValueError(f"Unsupported approach '{APPROACH_NAME}'.")


def main() -> None:
    output_dir = ensure_output_directory()
    output_files = build_output_filenames()
    names = load_json(DATA_DIR / "names.json")
    origins = load_json(DATA_DIR / "origins.json")

    structured_rows = build_structured_rows(names, origins)
    cleaned_rows = build_cleaned_rows(structured_rows)
    groups, normalized_to_group, diagnostics = run_selected_approach(cleaned_rows)
    final_dataset = build_final_dataset(cleaned_rows, normalized_to_group)

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
            "approach_name": APPROACH_NAME,
            "model_name": MODEL_NAME,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "context_similarity_threshold": CONTEXT_SIMILARITY_THRESHOLD,
            "batch_size": BATCH_SIZE,
            "diagnostics": diagnostics,
            "input_files": {
                "names": str(DATA_DIR / "names.json"),
                "origins": str(DATA_DIR / "origins.json"),
            },
            "output_dir": str(output_dir),
            "output_files": output_files,
        },
    )

    print(f"Created {output_dir / output_files['structured_csv']}")
    print(f"Created {output_dir / output_files['structured_json']}")
    print(f"Created {output_dir / output_files['cleaned_json']}")
    print(f"Created {output_dir / output_files['groups_json']}")
    print(f"Created {output_dir / output_files['final_json']}")
    print(f"Created {output_dir / output_files['metadata_json']}")


if __name__ == "__main__":
    main()
