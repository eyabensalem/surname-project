from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

ORIGINS_FILE = DATA_DIR / "origins.json"
GROUPED_NAMES_FILE = RESULTS_DIR / "final_dataset.json"
MERGED_GROUPS_FILE = RESULTS_DIR / "merged_groups.json"
GROUP_SUMMARIES_FILE = RESULTS_DIR / "group_summaries.json"

FIRSTNAMES_LIST_FILE = RESULTS_DIR / "firstnames_list.json"
FIRSTNAMES_DATASET_FILE = RESULTS_DIR / "firstnames_dataset.json"
FIRSTNAMES_SUMMARIES_FILE = RESULTS_DIR / "firstnames_summaries.json"
MODEL_COMPARISON_FILE = RESULTS_DIR / "model_comparison.json"
EVALUATION_RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"

MODEL_SCORES= RESULTS_DIR / "model_scores.png"
SURNAME_VARIANT_GRAPH = RESULTS_DIR / "surname_variant_graph.png"