import argparse
import json
import math
from collections import defaultdict
from html import escape
from math import comb
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"
OUTPUTS_DIR = TEST_DIR / "outputs"
VISUALS_DIR = OUTPUTS_DIR / "visualizations"

APPROACH_OUTPUTS = {
    "approach_1_name_similarity": OUTPUTS_DIR / "01_name_similarity" / "final_dataset_name_similarity.json",
    "approach_2_name_and_context": OUTPUTS_DIR / "02_name_and_context" / "final_dataset_name_and_context.json",
    "approach_3_sequence_matcher": OUTPUTS_DIR / "03_sequence_matcher" / "final_dataset_sequence_matcher.json",
    "approach_4_levenshtein": OUTPUTS_DIR / "04_levenshtein" / "final_dataset_levenshtein.json",
    "approach_5_soundex": OUTPUTS_DIR / "05_soundex" / "final_dataset_soundex.json",
    "approach_6_spacy": OUTPUTS_DIR / "06_spacy" / "final_dataset_spacy.json",
}
DEFAULT_APPROACHES = [
    "approach_2_name_and_context",
    "approach_3_sequence_matcher",
    "approach_4_levenshtein",
    "approach_5_soundex",
    "approach_6_spacy",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)


def write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)


def extract_variant_names(group: dict) -> list[str]:
    if "variants" in group:
        return group["variants"]
    if "variant_names" in group:
        return group["variant_names"]
    raise ValueError("Each group must contain either 'variants' or 'variant_names'.")


def build_gold_mapping(gold_groups: list[dict]) -> dict[str, str]:
    gold_mapping = {}
    for group in gold_groups:
        gold_group_id = group["gold_group_id"]
        for name in extract_variant_names(group):
            if name in gold_mapping:
                raise ValueError(f"Duplicate gold annotation for '{name}'.")
            gold_mapping[name] = gold_group_id
    return gold_mapping


def build_predicted_mapping(predicted_groups: list[dict]) -> dict[str, str]:
    predicted_mapping = {}
    for index, group in enumerate(predicted_groups, start=1):
        group_id = str(group.get("group_id", index))
        for name in extract_variant_names(group):
            predicted_mapping[name] = group_id
    return predicted_mapping


def build_contingency(names: list[str], gold_mapping: dict[str, str], predicted_mapping: dict[str, str]):
    contingency: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    predicted_sizes: defaultdict[str, int] = defaultdict(int)
    gold_sizes: defaultdict[str, int] = defaultdict(int)

    for name in names:
        gold_label = gold_mapping[name]
        predicted_label = predicted_mapping.get(name, f"__missing__:{name}")
        contingency[predicted_label][gold_label] += 1
        predicted_sizes[predicted_label] += 1
        gold_sizes[gold_label] += 1

    return contingency, predicted_sizes, gold_sizes


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_pairwise_metrics(gold_mapping: dict[str, str], predicted_mapping: dict[str, str]) -> dict[str, float]:
    names = sorted(gold_mapping)
    contingency, predicted_sizes, gold_sizes = build_contingency(names, gold_mapping, predicted_mapping)

    true_positive = sum(comb(count, 2) for row in contingency.values() for count in row.values() if count > 1)
    predicted_positive = sum(comb(size, 2) for size in predicted_sizes.values() if size > 1)
    gold_positive = sum(comb(size, 2) for size in gold_sizes.values() if size > 1)

    false_positive = predicted_positive - true_positive
    false_negative = gold_positive - true_positive

    precision = safe_divide(true_positive, predicted_positive)
    recall = safe_divide(true_positive, gold_positive)
    f1 = safe_divide(2 * precision * recall, precision + recall) if precision or recall else 0.0

    return {
        "evaluated_name_count": len(names),
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "false_negative_pairs": false_negative,
        "predicted_positive_pairs": predicted_positive,
        "gold_positive_pairs": gold_positive,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "false_merge_rate": round(safe_divide(false_positive, predicted_positive), 6),
        "false_split_rate": round(safe_divide(false_negative, gold_positive), 6),
    }


def format_metrics(label: str, metrics: dict[str, float]) -> str:
    return "\n".join(
        [
            label,
            f"  evaluated_name_count : {metrics['evaluated_name_count']}",
            f"  precision            : {metrics['precision']}",
            f"  recall               : {metrics['recall']}",
            f"  f1_score             : {metrics['f1_score']}",
            f"  false_merge_rate     : {metrics['false_merge_rate']}",
            f"  false_split_rate     : {metrics['false_split_rate']}",
            f"  true_positive_pairs  : {metrics['true_positive_pairs']}",
            f"  false_positive_pairs : {metrics['false_positive_pairs']}",
            f"  false_negative_pairs : {metrics['false_negative_pairs']}",
        ]
    )


def slugify(label: str) -> str:
    return label.replace("approach_", "").replace("_", "-")


def friendly_name(approach_name: str) -> str:
    mapping = {
        "approach_1_name_similarity": "Name Similarity",
        "approach_2_name_and_context": "Name And Context",
        "approach_3_sequence_matcher": "SequenceMatcher",
        "approach_4_levenshtein": "Levenshtein",
        "approach_5_soundex": "Soundex",
        "approach_6_spacy": "spaCy",
    }
    return mapping.get(approach_name, approach_name)


def format_score(value: float) -> str:
    return f"{value:.3f}"


def wrap_svg(content: str, width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">{content}</svg>'
    )


def build_horizontal_bar_chart(
    title: str,
    metric_key: str,
    summary: list[dict],
    color: str,
    subtitle: str,
) -> str:
    width = 900
    row_height = 72
    top = 110
    bottom = 50
    left = 240
    right = 80
    chart_width = width - left - right
    height = top + len(summary) * row_height + bottom

    lines = [
        '<rect width="100%" height="100%" fill="#f7f4ea"/>',
        '<rect x="24" y="24" width="852" height="{h}" rx="20" fill="#fffdf8" stroke="#d7cfbf"/>'.format(
            h=height - 48
        ),
        f'<text x="48" y="64" font-size="28" font-family="Segoe UI, Arial, sans-serif" fill="#1f2933">{escape(title)}</text>',
        f'<text x="48" y="90" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#5b6570">{escape(subtitle)}</text>',
    ]

    for tick in range(6):
        x = left + chart_width * tick / 5
        label = f"{tick / 5:.1f}"
        lines.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - bottom}" stroke="#e7e1d5"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" font-size="12" '
            'font-family="Segoe UI, Arial, sans-serif" fill="#7b8794">{}</text>'.format(label)
        )

    for index, item in enumerate(summary):
        metric_value = item["metrics"][metric_key]
        y = top + index * row_height
        bar_width = chart_width * metric_value
        label = friendly_name(item["approach_name"])

        lines.append(
            f'<text x="{left - 16}" y="{y + 24}" text-anchor="end" font-size="16" '
            'font-family="Segoe UI, Arial, sans-serif" fill="#243b53">{}</text>'.format(escape(label))
        )
        lines.append(
            f'<rect x="{left}" y="{y + 6}" width="{chart_width}" height="24" rx="12" fill="#ece7dc"/>'
        )
        lines.append(
            f'<rect x="{left}" y="{y + 6}" width="{bar_width:.1f}" height="24" rx="12" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{left + chart_width + 12}" y="{y + 24}" font-size="15" '
            'font-family="Consolas, Courier New, monospace" fill="#102a43">{}</text>'.format(format_score(metric_value))
        )

    return wrap_svg("".join(lines), width, height)


def build_grouped_metric_chart(summary: list[dict]) -> str:
    width = 980
    height = 520
    top = 110
    bottom = 70
    left = 90
    right = 40
    chart_height = height - top - bottom
    cluster_width = (width - left - right) / max(len(summary), 1)
    bar_width = min(52, cluster_width / 4)
    colors = {
        "precision": "#0f766e",
        "recall": "#c2410c",
        "f1_score": "#1d4ed8",
    }

    lines = [
        '<rect width="100%" height="100%" fill="#f5efe4"/>',
        '<rect x="22" y="22" width="936" height="476" rx="20" fill="#fffdf8" stroke="#d9cfbd"/>',
        '<text x="44" y="60" font-size="28" font-family="Segoe UI, Arial, sans-serif" fill="#1f2933">Precision / Recall / F1</text>',
        '<text x="44" y="88" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#5b6570">Grouped view to compare trade-offs between kept models.</text>',
    ]

    for tick in range(6):
        value = tick / 5
        y = top + chart_height - chart_height * value
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#ece4d6"/>')
        lines.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" '
            'font-family="Segoe UI, Arial, sans-serif" fill="#7b8794">{}</text>'.format(f"{value:.1f}")
        )

    legend_x = width - 250
    for index, (metric_key, color) in enumerate(colors.items()):
        y = 56 + index * 24
        lines.append(f'<rect x="{legend_x}" y="{y - 11}" width="14" height="14" rx="3" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x + 22}" y="{y}" font-size="13" font-family="Segoe UI, Arial, sans-serif" fill="#334e68">{escape(metric_key)}</text>'
        )

    for index, item in enumerate(summary):
        center_x = left + cluster_width * index + cluster_width / 2
        label = friendly_name(item["approach_name"])
        for bar_index, metric_key in enumerate(colors):
            metric_value = item["metrics"][metric_key]
            x = center_x + (bar_index - 1) * (bar_width + 8) - bar_width / 2
            bar_height = chart_height * metric_value
            y = top + chart_height - bar_height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="8" fill="{colors[metric_key]}"/>'
            )
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="11" '
                'font-family="Consolas, Courier New, monospace" fill="#102a43">{}</text>'.format(format_score(metric_value))
            )

        lines.append(
            f'<text x="{center_x:.1f}" y="{height - 28}" text-anchor="middle" font-size="13" '
            'font-family="Segoe UI, Arial, sans-serif" fill="#243b53">{}</text>'.format(escape(label))
        )

    return wrap_svg("".join(lines), width, height)


def build_radar_chart(summary: list[dict]) -> str:
    width = 980
    height = 680
    center_x = 490
    center_y = 360
    radius = 210
    metrics = ["precision", "recall", "f1_score", "false_merge_rate", "false_split_rate"]
    labels = ["Precision", "Recall", "F1", "False Merge", "False Split"]
    colors = ["#1d4ed8", "#c2410c", "#0f766e"]

    lines = [
        '<rect width="100%" height="100%" fill="#f7f3eb"/>',
        '<rect x="22" y="22" width="936" height="636" rx="20" fill="#fffdf8" stroke="#d9cfbd"/>',
        '<text x="44" y="60" font-size="28" font-family="Segoe UI, Arial, sans-serif" fill="#1f2933">Error Profile Radar</text>',
        '<text x="44" y="88" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#5b6570">For false rates, lower is better. Outer area means larger value.</text>',
    ]

    point_cache = []
    for axis_index, label in enumerate(labels):
        angle = (-90 + axis_index * 360 / len(labels)) * math.pi / 180
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        point_cache.append((x, y, angle))

    for level in range(1, 6):
        level_radius = radius * level / 5
        polygon_points = []
        for _, _, angle in point_cache:
            x = center_x + level_radius * math.cos(angle)
            y = center_y + level_radius * math.sin(angle)
            polygon_points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            '<polygon points="{}" fill="none" stroke="#e8dfd0" stroke-width="1"/>'.format(" ".join(polygon_points))
        )
        lines.append(
            f'<text x="{center_x + 6}" y="{center_y - level_radius + 4:.1f}" font-size="11" '
            'font-family="Segoe UI, Arial, sans-serif" fill="#7b8794">{}</text>'.format(f"{level / 5:.1f}")
        )

    for (x, y, _), label in zip(point_cache, labels, strict=True):
        lines.append(f'<line x1="{center_x}" y1="{center_y}" x2="{x:.1f}" y2="{y:.1f}" stroke="#ddd3c2"/>')
        anchor = "middle"
        if x < center_x - 10:
            anchor = "end"
        elif x > center_x + 10:
            anchor = "start"
        lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="14" '
            'font-family="Segoe UI, Arial, sans-serif" fill="#243b53">{}</text>'.format(escape(label))
        )

    for index, item in enumerate(summary):
        points = []
        for metric_key, (_, _, angle) in zip(metrics, point_cache, strict=True):
            value = item["metrics"][metric_key]
            plot_value = 1 - value if metric_key.startswith("false_") else value
            x = center_x + radius * plot_value * math.cos(angle)
            y = center_y + radius * plot_value * math.sin(angle)
            points.append((x, y))
        point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        color = colors[index % len(colors)]
        lines.append(
            f'<polygon points="{point_string}" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="3"/>'
        )
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')

    legend_x = 740
    for index, item in enumerate(summary):
        y = 150 + index * 28
        color = colors[index % len(colors)]
        lines.append(f'<rect x="{legend_x}" y="{y - 11}" width="14" height="14" rx="3" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x + 22}" y="{y}" font-size="13" font-family="Segoe UI, Arial, sans-serif" fill="#334e68">{escape(friendly_name(item["approach_name"]))}</text>'
        )

    return wrap_svg("".join(lines), width, height)


def build_dashboard_html(summary: list[dict]) -> str:
    rows = []
    for item in summary:
        metrics = item["metrics"]
        rows.append(
            "<tr>"
            f"<td>{escape(friendly_name(item['approach_name']))}</td>"
            f"<td>{format_score(metrics['precision'])}</td>"
            f"<td>{format_score(metrics['recall'])}</td>"
            f"<td>{format_score(metrics['f1_score'])}</td>"
            f"<td>{format_score(metrics['false_merge_rate'])}</td>"
            f"<td>{format_score(metrics['false_split_rate'])}</td>"
            f"<td>{metrics['true_positive_pairs']}</td>"
            f"<td>{metrics['false_positive_pairs']}</td>"
            f"<td>{metrics['false_negative_pairs']}</td>"
            "</tr>"
        )

    cards = []
    for item in summary:
        metrics = item["metrics"]
        cards.append(
            '<article class="card">'
            f"<h3>{escape(friendly_name(item['approach_name']))}</h3>"
            f"<p class=\"score\">F1 {format_score(metrics['f1_score'])}</p>"
            f"<p>Precision {format_score(metrics['precision'])} · Recall {format_score(metrics['recall'])}</p>"
            f"<p>False merge {format_score(metrics['false_merge_rate'])} · False split {format_score(metrics['false_split_rate'])}</p>"
            "</article>"
        )

    html = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comparison Dashboard</title>
  <style>
    :root {
      --bg: #f5efe4;
      --panel: #fffdf8;
      --line: #d9cfbd;
      --text: #1f2933;
      --muted: #5b6570;
      --accent: #1d4ed8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: radial-gradient(circle at top left, #fff8ec, var(--bg) 55%);
      color: var(--text);
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 8px 24px rgba(31, 41, 51, 0.06);
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 34px; }
    .sub { margin-top: 8px; color: var(--muted); }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      background: linear-gradient(180deg, #fffefb, #f8f2e8);
    }
    .card .score {
      margin: 8px 0;
      font-size: 28px;
      font-weight: 700;
      color: var(--accent);
    }
    img {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 14px;
    }
    th, td {
      border-bottom: 1px solid #eadfce;
      padding: 10px 8px;
      text-align: left;
    }
    th {
      color: var(--muted);
      font-weight: 600;
    }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Model Comparison Dashboard</h1>
      <p class="sub">Visual summary for the selected grouping models on the test gold set.</p>
      <div class="cards">__CARDS__</div>
    </section>
    <section>
      <h2>F1 Ranking</h2>
      <img src="comparison_f1.svg" alt="F1 ranking chart">
    </section>
    <section>
      <h2>Precision, Recall and F1</h2>
      <img src="comparison_prf.svg" alt="Precision recall f1 grouped chart">
    </section>
    <section>
      <h2>Error Profile</h2>
      <img src="comparison_radar.svg" alt="Radar chart for error profile">
    </section>
    <section>
      <h2>Metrics Table</h2>
      <table>
        <thead>
          <tr>
            <th>Approach</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>False merge</th>
            <th>False split</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
          </tr>
        </thead>
        <tbody>__ROWS__</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""
    return html.replace("__CARDS__", "".join(cards)).replace("__ROWS__", "".join(rows))


def write_visualizations(summary: list[dict]) -> list[str]:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "comparison_f1.svg": build_horizontal_bar_chart(
            title="F1 Ranking",
            metric_key="f1_score",
            summary=summary,
            color="#1d4ed8",
            subtitle="Main ranking metric across selected models.",
        ),
        "comparison_prf.svg": build_grouped_metric_chart(summary),
        "comparison_radar.svg": build_radar_chart(summary),
        "comparison_dashboard.html": build_dashboard_html(summary),
    }
    for filename, content in outputs.items():
        write_text(VISUALS_DIR / filename, content)
    return [str(VISUALS_DIR / filename) for filename in outputs]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the selected test approaches against the gold file."
    )
    parser.add_argument(
        "--gold",
        default=str(DATA_DIR / "gold_clusters.template.json"),
        help="Path to the gold clusters JSON file.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=DEFAULT_APPROACHES,
        help="Approach names to compare.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "comparison_summary.json"),
        help="Where to write the JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    invalid = [name for name in args.approaches if name not in APPROACH_OUTPUTS]
    if invalid:
        valid_values = ", ".join(sorted(APPROACH_OUTPUTS))
        raise ValueError(f"Unknown approach(es): {', '.join(invalid)}. Expected values: {valid_values}")

    gold_mapping = build_gold_mapping(load_json(gold_path))
    summary = []

    for approach_name in args.approaches:
        prediction_path = APPROACH_OUTPUTS[approach_name]
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

        metrics = compute_pairwise_metrics(gold_mapping, build_predicted_mapping(load_json(prediction_path)))
        summary.append(
            {
                "approach_name": approach_name,
                "prediction_file": str(prediction_path),
                "metrics": metrics,
            }
        )

    summary.sort(key=lambda item: item["metrics"]["f1_score"], reverse=True)

    for item in summary:
        print(format_metrics(item["approach_name"], item["metrics"]))
        print()

    write_json(Path(args.output), summary)
    print(f"Created {args.output}")
    visual_files = write_visualizations(summary)
    for visual_file in visual_files:
        print(f"Created {visual_file}")


if __name__ == "__main__":
    main()
