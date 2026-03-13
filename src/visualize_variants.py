import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # use non-interactive backend

import matplotlib.pyplot as plt
import networkx as nx


from config import GROUPED_NAMES_FILE, SURNAME_VARIANT_GRAPH

INPUT_FILE = GROUPED_NAMES_FILE
OUTPUT_FILE = SURNAME_VARIANT_GRAPH

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    groups = json.load(f)

G = nx.Graph()

for group in groups:
    variants = group.get("variants", [])
    if not variants:
        continue

    main_variant = variants[0]
    for variant in variants[1:]:
        G.add_edge(main_variant, variant)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=1500, font_size=8)
plt.title("Surname Variant Graph")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Graph saved to: {OUTPUT_FILE}")