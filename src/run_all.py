"""
Run the full NLP pipeline.
"""

import subprocess
import sys


def run_script(script_name):
    print(f"\nRunning {script_name} ...")
    subprocess.run([sys.executable, f"src/{script_name}"], check=True)


def main():

    pipeline = [
        "text_grouping.py",
        "summarization.py",
        "compare_summarizers.py",
        "evaluate_summaries.py",
        "visualize_variants.py",
        "plot_model_scores.py",
        "scrape_firstname_list.py",
        "scrape_firstname_details.py",
        "summarize_firstnames.py",
    ]

    for script in pipeline:
        run_script(script)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()