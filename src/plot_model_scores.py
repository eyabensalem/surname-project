import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # use non-interactive backend

import matplotlib.pyplot as plt

from config import EVALUATION_RESULTS_FILE,MODEL_SCORES

INPUT_FILE = EVALUATION_RESULTS_FILE
OUTPUT_FILE = MODEL_SCORES




with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

models = ["tfidf_scores", "textrank_scores", "bart_scores"]
avg_scores = {}

for model in models:
    rouge1_vals = [item[model]["rouge1_f1"] for item in data]
    avg_scores[model] = sum(rouge1_vals) / len(rouge1_vals)

plt.figure(figsize=(8, 5))
plt.bar(avg_scores.keys(), avg_scores.values())
plt.title("Average ROUGE-1 F1 by Model")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Chart saved to: {OUTPUT_FILE}")