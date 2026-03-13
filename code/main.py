import csv
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUTS_DIR = BASE_DIR.parent / "outputs"
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
APPROACH_NAME = os.getenv("NAME_GROUP_APPROACH", "approach_1_name_similarity")
MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
SPACY_MODEL = os.getenv("SPACY_MODEL", "fr_core_news_md")
SIMILARITY_THRESHOLD = float(os.getenv("NAME_GROUP_THRESHOLD", "0.78"))
CONTEXT_SIMILARITY_THRESHOLD = float(os.getenv("NAME_CONTEXT_GROUP_THRESHOLD", str(SIMILARITY_THRESHOLD)))
SEQUENCE_MATCHER_THRESHOLD = float(os.getenv("SEQUENCE_MATCHER_THRESHOLD", "0.88"))
LEVENSHTEIN_THRESHOLD = float(os.getenv("LEVENSHTEIN_THRESHOLD", "0.85"))
SOUNDEX_THRESHOLD = float(os.getenv("SOUNDEX_THRESHOLD", "1.0"))
SPACY_SIMILARITY_THRESHOLD = float(os.getenv("SPACY_SIMILARITY_THRESHOLD", "0.75"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
SUMMARY_TOP_N = int(os.getenv("GROUP_SUMMARY_SENTENCE_COUNT", "2"))

IMPORTANT_SUMMARY_KEYWORDS = [
    "signifie",
    "designe",
    "variante",
    "variantes",
    "forme",
    "formes",
    "derive",
    "derives",
    "origine",
    "originaire",
    "toponyme",
    "prenom",
    "nom",
]


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
        "merged_groups_json": f"merged_groups_{suffix}.json",
        "summaries_json": f"group_summaries_{suffix}.json",
        "metadata_json": f"run_metadata_{suffix}.json",
    }


def normalize_text(text: str) -> str:
    text = unidecode(text.lower())
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
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


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


def build_merged_groups(final_dataset: list[dict], origins: dict[str, str]) -> list[dict]:
    merged_groups = []
    for group in final_dataset:
        origin_ids = group.get("origin_ids", [])
        texts = [origins[origin_id] for origin_id in origin_ids if origins.get(origin_id)]
        merged_groups.append(
            {
                "group_id": group.get("group_id"),
                "variants": group.get("variants", []),
                "origin_ids": origin_ids,
                "origin_texts": texts,
                "merged_text": " ".join(texts),
            }
        )
    return merged_groups


def compute_keyword_score(sentence: str) -> float:
    lowered = normalize_text(sentence)
    score = 0.0

    for keyword in IMPORTANT_SUMMARY_KEYWORDS:
        if keyword in lowered:
            score += 2.0

    if "signifie" in lowered:
        score += 6.0
    if "variante" in lowered or "variantes" in lowered:
        score += 2.5
    if "designe" in lowered:
        score += 2.5
    if lowered.startswith("le nom signifie"):
        score += 4.0
    if "prenom" in lowered:
        score += 2.0

    return score


def compute_tfidf_scores(sentences: list[str]) -> np.ndarray:
    if not sentences:
        return np.array([])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(sentences)
    return np.asarray(matrix.sum(axis=1)).ravel()


def rank_sentences(sentences: list[str]) -> list[tuple[int, float]]:
    if not sentences:
        return []

    tfidf_scores = compute_tfidf_scores(sentences)
    if len(tfidf_scores) > 0 and tfidf_scores.max() > 0:
        normalized_tfidf = tfidf_scores / tfidf_scores.max()
    else:
        normalized_tfidf = tfidf_scores

    ranked = []
    for index, sentence in enumerate(sentences):
        hybrid_score = compute_keyword_score(sentence) + float(normalized_tfidf[index])
        if index == 0:
            hybrid_score += 1.5
        if index == 1:
            hybrid_score += 0.5

        length = max(len(sentence.split()), 1)
        hybrid_score -= length / 100
        ranked.append((index, hybrid_score))

    return ranked


def compute_summary_confidence(sentences: list[str], ranked: list[tuple[int, float]], top_n: int) -> float:
    if not sentences or not ranked:
        return 0.0

    best_scores = sorted((score for _, score in ranked), reverse=True)[:top_n]
    if not best_scores:
        return 0.0

    confidence = sum(best_scores) / len(best_scores)
    return round(min(confidence / 10, 1.0), 3)


def summarize_text(text: str, top_n: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= top_n:
        return clean_summary_text(" ".join(sentences))

    ranked = rank_sentences(sentences)
    top_indices = sorted(index for index, _ in sorted(ranked, key=lambda item: item[1], reverse=True)[:top_n])
    return clean_summary_text(" ".join(sentences[index] for index in top_indices))


def build_group_summaries(merged_groups: list[dict], top_n: int) -> list[dict]:
    summaries = []
    for group in merged_groups:
        merged_text = group.get("merged_text", "")
        sentences = split_sentences(merged_text)
        ranked = rank_sentences(sentences)
        summaries.append(
            {
                "group_id": group.get("group_id"),
                "variants": group.get("variants", []),
                "origin_ids": group.get("origin_ids", []),
                "summary": summarize_text(merged_text, top_n=top_n),
                "confidence_score": compute_summary_confidence(sentences, ranked, top_n=top_n),
            }
        )

    return summaries


def run_selected_approach(cleaned_rows: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    if APPROACH_NAME == "approach_1_name_similarity":
        return build_groups_from_embeddings(
            cleaned_rows,
            comparison_text_builder=build_name_only_text,
            similarity_threshold=SIMILARITY_THRESHOLD,
            approach_label=APPROACH_NAME,
        )

    if APPROACH_NAME == "approach_2_name_and_context":
        return build_groups_from_embeddings(
            cleaned_rows,
            comparison_text_builder=build_name_and_context_text,
            similarity_threshold=CONTEXT_SIMILARITY_THRESHOLD,
            approach_label=APPROACH_NAME,
        )

    if APPROACH_NAME == "approach_3_sequence_matcher":
        return build_groups_with_sequence_matcher(cleaned_rows)

    if APPROACH_NAME == "approach_4_levenshtein":
        return build_groups_with_levenshtein(cleaned_rows)

    if APPROACH_NAME == "approach_5_soundex":
        return build_groups_with_soundex(cleaned_rows)

    if APPROACH_NAME == "approach_6_spacy":
        return build_groups_with_spacy(cleaned_rows)

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
    merged_groups = build_merged_groups(final_dataset, origins)
    group_summaries = build_group_summaries(merged_groups, top_n=SUMMARY_TOP_N)

    write_csv(
        output_dir / output_files["structured_csv"],
        structured_rows,
        ["name", "origin_id", "origin_text"],
    )
    write_json(output_dir / output_files["structured_json"], structured_rows)
    write_json(output_dir / output_files["cleaned_json"], cleaned_rows)
    write_json(output_dir / output_files["groups_json"], groups)
    write_json(output_dir / output_files["final_json"], final_dataset)
    write_json(output_dir / output_files["merged_groups_json"], merged_groups)
    write_json(output_dir / output_files["summaries_json"], group_summaries)
    write_json(
        output_dir / output_files["metadata_json"],
        {
            "approach_name": APPROACH_NAME,
            "model_name": MODEL_NAME,
            "spacy_model": SPACY_MODEL,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "context_similarity_threshold": CONTEXT_SIMILARITY_THRESHOLD,
            "sequence_matcher_threshold": SEQUENCE_MATCHER_THRESHOLD,
            "levenshtein_threshold": LEVENSHTEIN_THRESHOLD,
            "soundex_threshold": SOUNDEX_THRESHOLD,
            "spacy_similarity_threshold": SPACY_SIMILARITY_THRESHOLD,
            "batch_size": BATCH_SIZE,
            "group_summary_sentence_count": SUMMARY_TOP_N,
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
    print(f"Created {output_dir / output_files['merged_groups_json']}")
    print(f"Created {output_dir / output_files['summaries_json']}")
    print(f"Created {output_dir / output_files['metadata_json']}")


if __name__ == "__main__":
    main()
