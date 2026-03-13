"""
Run the full NLP pipeline.
"""

import os
import shutil
import subprocess
import sys

from config import (
    GROUPED_NAMES_FILE,
    GROUP_SUMMARIES_FILE,
    MERGED_GROUPS_FILE,
    PRIMARY_APPROACH_NAME,
    PRIMARY_FINAL_DATASET_FILE,
    PRIMARY_GROUP_SUMMARIES_FILE,
    PRIMARY_MERGED_GROUPS_FILE,
    RESULTS_DIR,
)


def run_command(label, command, env=None):
    print(f"\nRunning {label} ...")
    subprocess.run(command, check=True, env=env)


def run_script(script_name):
    run_command(script_name, [sys.executable, f"src/{script_name}"])


def run_primary_grouping_pipeline():
    env = os.environ.copy()
    env["NAME_GROUP_APPROACH"] = PRIMARY_APPROACH_NAME
    run_command(
        f"code/main.py ({PRIMARY_APPROACH_NAME})",
        [sys.executable, "code/main.py"],
        env=env,
    )


def synchronize_surname_artifacts():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = [
        (PRIMARY_FINAL_DATASET_FILE, GROUPED_NAMES_FILE),
        (PRIMARY_MERGED_GROUPS_FILE, MERGED_GROUPS_FILE),
        (PRIMARY_GROUP_SUMMARIES_FILE, GROUP_SUMMARIES_FILE),
    ]

    for source_path, target_path in artifacts:
        if not source_path.exists():
            raise FileNotFoundError(f"Required source file not found: {source_path}")

        shutil.copy2(source_path, target_path)
        print(f"Synchronized {source_path.name} -> {target_path}")


def main():
    run_primary_grouping_pipeline()
    synchronize_surname_artifacts()

    pipeline = [
        "compare_summarizers.py",
        "evaluate_summaries.py",
        "plot_model_scores.py",
        "scrape_firstname_list.py",
        "scrape_firstname_details.py",
        "summarize_firstnames.py",
        "group_firstnames_soundex.py",
    ]

    for script in pipeline:
        run_script(script)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
