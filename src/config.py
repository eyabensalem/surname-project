import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

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
PRIMARY_APPROACH_NAME = os.getenv("APP_SURNAME_APPROACH", "approach_5_soundex")

try:
    PRIMARY_APPROACH_DIRECTORY = APPROACH_DIRECTORIES[PRIMARY_APPROACH_NAME]
    PRIMARY_APPROACH_SUFFIX = APPROACH_FILE_SUFFIXES[PRIMARY_APPROACH_NAME]
except KeyError as exc:
    valid_values = ", ".join(sorted(APPROACH_DIRECTORIES))
    raise ValueError(
        f"Unknown APP_SURNAME_APPROACH '{PRIMARY_APPROACH_NAME}'. Expected one of: {valid_values}"
    ) from exc

PRIMARY_SURNAME_OUTPUT_DIR = BASE_DIR / "outputs" / PRIMARY_APPROACH_DIRECTORY
PRIMARY_FINAL_DATASET_FILE = PRIMARY_SURNAME_OUTPUT_DIR / f"final_dataset_{PRIMARY_APPROACH_SUFFIX}.json"
PRIMARY_MERGED_GROUPS_FILE = PRIMARY_SURNAME_OUTPUT_DIR / f"merged_groups_{PRIMARY_APPROACH_SUFFIX}.json"
PRIMARY_GROUP_SUMMARIES_FILE = PRIMARY_SURNAME_OUTPUT_DIR / f"group_summaries_{PRIMARY_APPROACH_SUFFIX}.json"

ORIGINS_FILE = DATA_DIR / "origins.json"
GROUPED_NAMES_FILE = RESULTS_DIR / "final_dataset.json"
MERGED_GROUPS_FILE = RESULTS_DIR / "merged_groups.json"
GROUP_SUMMARIES_FILE = RESULTS_DIR / "group_summaries.json"

FIRSTNAMES_LIST_FILE = RESULTS_DIR / "firstnames_list.json"
FIRSTNAMES_DATASET_FILE = RESULTS_DIR / "firstnames_dataset.json"
FIRSTNAMES_SUMMARIES_FILE = RESULTS_DIR / "firstnames_summaries.json"
FIRSTNAMES_GROUPED_FILE = RESULTS_DIR / "firstnames_grouped_soundex.json"
FIRSTNAMES_GROUP_SUMMARIES_FILE = RESULTS_DIR / "firstnames_group_summaries_soundex.json"
MODEL_COMPARISON_FILE = RESULTS_DIR / "model_comparison.json"
EVALUATION_RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"

MODEL_SCORES= RESULTS_DIR / "model_scores.png"
