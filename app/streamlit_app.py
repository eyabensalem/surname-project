import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
import unicodedata
import streamlit as st
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
TEST_OUTPUTS_DIR = BASE_DIR / "test" / "outputs"
TEST_VISUALS_DIR = TEST_OUTPUTS_DIR / "visualizations"
PRIMARY_SURNAME_SUMMARIES_FILE = RESULTS_DIR / "group_summaries.json"
FALLBACK_SURNAME_SUMMARIES_FILE = BASE_DIR / "outputs" / "05_soundex" / "group_summaries_soundex.json"
COMPARISON_SUMMARY_FILE = TEST_OUTPUTS_DIR / "comparison_summary.json"
PRIMARY_FIRSTNAME_GROUP_SUMMARIES_FILE = RESULTS_DIR / "firstnames_group_summaries_soundex.json"
LEGACY_FIRSTNAME_SUMMARIES_FILE = RESULTS_DIR / "firstnames_summaries.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_existing_file(*paths: Path):
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def soundex(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    letters = "".join(char for char in normalized if char.isalpha())
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


# Load data
surname_summaries_file = resolve_existing_file(
    PRIMARY_SURNAME_SUMMARIES_FILE,
    FALLBACK_SURNAME_SUMMARIES_FILE,
)
if surname_summaries_file is None:
    raise FileNotFoundError(
        "No surname summaries found. Run the pipeline first or provide group summaries in results/."
    )

group_summaries = load_json(surname_summaries_file)


# Load full firstname dataset for visualizations
firstname_dataset = []
dataset_file = RESULTS_DIR / "firstnames_dataset.json"

if dataset_file.exists():
    firstname_dataset = load_json(dataset_file)

# Build list of surname variants
surname_variant_rows = []

for item in group_summaries:
    for variant in item.get("variants", []):
        surname_variant_rows.append(
            {
                "variant": variant,
                "variant_norm": normalize_text(variant),
                "group_data": item,
            }
        )
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()
firstname_summaries = []
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()
@st.cache_resource
def build_surname_embeddings(rows):
    texts = [row["variant_norm"] for row in rows]
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)
    return embeddings


surname_embeddings = build_surname_embeddings(surname_variant_rows)
firstname_group_summaries = []
firstname_group_file = resolve_existing_file(PRIMARY_FIRSTNAME_GROUP_SUMMARIES_FILE)
if firstname_group_file is not None:
    firstname_group_summaries = load_json(firstname_group_file)

firstname_file = LEGACY_FIRSTNAME_SUMMARIES_FILE
if firstname_file.exists():
    firstname_summaries = load_json(firstname_file)

comparison_summary = []
if COMPARISON_SUMMARY_FILE.exists():
    comparison_summary = load_json(COMPARISON_SUMMARY_FILE)

st.set_page_config(
    page_title="Origines des Noms",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Injected CSS & SVG tree ──────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&display=swap');

  /* ── Reset & palette ── */
  :root {
    --parchment: #f5edd6;
    --parchment-dark: #e8d9b5;
    --ink: #2c1a0e;
    --ink-light: #5a3e28;
    --gold: #b8922a;
    --gold-light: #d4ae55;
    --forest: #2d4a2d;
    --forest-light: #3d6b3d;
    --rust: #7a3020;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--parchment) !important;
    background-image:
      url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Crect width='4' height='4' fill='%23f5edd6'/%3E%3Ccircle cx='1' cy='1' r='0.6' fill='%23e0ceaa' opacity='0.4'/%3E%3C/svg%3E");
    font-family: 'Crimson Pro', Georgia, serif;
    color: var(--ink);
  }

  [data-testid="stHeader"] { background: transparent !important; }

  /* ── Hero banner ── */
  .hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    position: relative;
  }
  .hero-title {
    font-family: 'Cinzel Decorative', serif;
    font-size: clamp(1.8rem, 4vw, 3rem);
    color: var(--ink);
    letter-spacing: 0.06em;
    line-height: 1.2;
    margin: 0;
    text-shadow: 1px 2px 0 var(--parchment-dark);
  }
  .hero-subtitle {
    font-family: 'Crimson Pro', serif;
    font-size: 1.15rem;
    font-style: italic;
    color: var(--ink-light);
    margin-top: 0.4rem;
    letter-spacing: 0.08em;
  }
  .divider-ornament {
    margin: 1rem auto;
    display: block;
    width: 340px;
    max-width: 90%;
  }

  /* ── SVG Tree container ── */
  .tree-wrap {
    display: flex;
    justify-content: center;
    margin: 0.5rem auto 2rem;
  }
  .tree-wrap svg {
    max-width: 100%;
    height: auto;
    filter: drop-shadow(0 4px 16px rgba(44,26,14,0.18));
  }

  /* ── Tabs ── */
  [data-testid="stTabs"] button {
    font-family: 'Cinzel Decorative', serif !important;
    font-size: 0.78rem !important;
    color: var(--ink-light) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    letter-spacing: 0.05em;
    transition: color 0.2s, border-color 0.2s;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
  }
  [data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--parchment-dark) !important;
    gap: 1.5rem;
  }

  /* ── Text input ── */
  [data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.55) !important;
    border: 1.5px solid var(--gold) !important;
    border-radius: 4px !important;
    font-family: 'Crimson Pro', serif !important;
    font-size: 1.1rem !important;
    color: var(--ink) !important;
    padding: 0.5rem 1rem !important;
    box-shadow: inset 0 1px 3px rgba(44,26,14,0.08);
    transition: box-shadow 0.2s;
  }
  [data-testid="stTextInput"] input:focus {
    box-shadow: 0 0 0 3px rgba(184,146,42,0.22) !important;
    outline: none !important;
  }
  [data-testid="stTextInput"] label {
    font-family: 'Cinzel Decorative', serif !important;
    font-size: 0.8rem !important;
    color: var(--ink-light) !important;
    letter-spacing: 0.05em;
  }

  /* ── Result card ── */
  .result-card {
    background: rgba(255,255,255,0.5);
    border: 1px solid var(--gold);
    border-left: 5px solid var(--gold);
    border-radius: 6px;
    padding: 1.4rem 1.8rem;
    margin-top: 1.2rem;
    position: relative;
    box-shadow: 0 2px 12px rgba(44,26,14,0.10);
  }
  .result-card::before {
    content: '⚜';
    position: absolute;
    top: -0.85rem;
    left: 1.2rem;
    font-size: 1.4rem;
    color: var(--gold);
    background: var(--parchment);
    padding: 0 0.3rem;
    line-height: 1;
  }
  .result-card h3 {
    font-family: 'Cinzel Decorative', serif;
    font-size: 1rem;
    color: var(--ink);
    margin: 0 0 0.6rem;
    letter-spacing: 0.05em;
  }
  .variant-pill {
    display: inline-block;
    background: var(--forest);
    color: #d4e8d4;
    font-size: 0.8rem;
    font-family: 'Crimson Pro', serif;
    letter-spacing: 0.04em;
    padding: 0.18rem 0.7rem;
    border-radius: 20px;
    margin: 0.15rem 0.2rem;
  }
  .score-badge {
    display: inline-block;
    background: var(--gold);
    color: var(--parchment);
    font-family: 'Cinzel Decorative', serif;
    font-size: 0.75rem;
    padding: 0.2rem 0.8rem;
    border-radius: 3px;
    margin-top: 0.7rem;
    letter-spacing: 0.04em;
  }
  .summary-text {
    font-size: 1.08rem;
    line-height: 1.7;
    color: var(--ink);
    margin-top: 0.6rem;
    font-style: italic;
  }

  /* ── Warning ── */
  [data-testid="stAlert"] {
    background: rgba(122,48,32,0.08) !important;
    border-left: 4px solid var(--rust) !important;
    border-radius: 4px !important;
    color: var(--rust) !important;
    font-family: 'Crimson Pro', serif !important;
    font-style: italic;
  }

  /* ── Viz images ── */
  [data-testid="stImage"] img {
    border-radius: 6px;
    border: 1px solid var(--parchment-dark);
    box-shadow: 0 3px 14px rgba(44,26,14,0.13);
  }

  /* ── Section headers (inside tabs) ── */
  .tab-header {
    font-family: 'Cinzel Decorative', serif;
    font-size: 1rem;
    color: var(--ink-light);
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--gold-light);
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
  }
</style>

<!-- ── Hero ── -->
<div class="hero">
  <p class="hero-title">Origines des Noms de Famille</p>
  <p class="hero-subtitle">Explorez l'histoire et la généalogie de votre patronyme</p>

  <!-- Ornamental divider -->
  <svg class="divider-ornament" viewBox="0 0 340 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <line x1="0" y1="12" x2="130" y2="12" stroke="#b8922a" stroke-width="1"/>
    <polygon points="155,2 165,12 155,22 145,12" fill="none" stroke="#b8922a" stroke-width="1.2"/>
    <circle cx="155" cy="12" r="3" fill="#b8922a"/>
    <polygon points="185,5 190,12 185,19 180,12" fill="none" stroke="#b8922a" stroke-width="1" opacity="0.6"/>
    <polygon points="125,5 130,12 125,19 120,12" fill="none" stroke="#b8922a" stroke-width="1" opacity="0.6"/>
    <line x1="180" y1="12" x2="340" y2="12" stroke="#b8922a" stroke-width="1"/>
  </svg>
</div>

<!-- ── Decorative Family Tree SVG ── -->
<div class="tree-wrap">
<svg width="640" height="320" viewBox="0 0 640 320" xmlns="http://www.w3.org/2000/svg">
  <!-- Ground -->
  <ellipse cx="320" cy="300" rx="180" ry="18" fill="#2d4a2d" opacity="0.18"/>

  <!-- Trunk -->
  <path d="M300 290 Q308 240 312 200 Q316 160 320 130" stroke="#5a3e28" stroke-width="18" fill="none" stroke-linecap="round"/>
  <path d="M340 290 Q332 240 328 200 Q324 160 320 130" stroke="#7a5c3a" stroke-width="10" fill="none" stroke-linecap="round"/>
  <!-- Trunk texture -->
  <path d="M310 270 Q316 255 312 240" stroke="#4a3020" stroke-width="2" fill="none" opacity="0.4"/>
  <path d="M318 280 Q322 262 320 248" stroke="#4a3020" stroke-width="2" fill="none" opacity="0.35"/>

  <!-- Roots -->
  <path d="M300 290 Q280 295 255 305" stroke="#5a3e28" stroke-width="8" fill="none" stroke-linecap="round"/>
  <path d="M255 305 Q235 310 215 308" stroke="#5a3e28" stroke-width="5" fill="none" stroke-linecap="round"/>
  <path d="M340 290 Q360 296 385 306" stroke="#5a3e28" stroke-width="8" fill="none" stroke-linecap="round"/>
  <path d="M385 306 Q405 311 425 308" stroke="#5a3e28" stroke-width="5" fill="none" stroke-linecap="round"/>
  <path d="M318 292 Q318 300 316 312" stroke="#5a3e28" stroke-width="6" fill="none" stroke-linecap="round"/>

  <!-- Main branches -->
  <!-- Left big -->
  <path d="M315 200 Q280 175 240 155" stroke="#5a3e28" stroke-width="9" fill="none" stroke-linecap="round"/>
  <path d="M240 155 Q210 140 180 135" stroke="#5a3e28" stroke-width="6" fill="none" stroke-linecap="round"/>
  <!-- Left sub -->
  <path d="M240 155 Q225 130 215 105" stroke="#6b4a30" stroke-width="5" fill="none" stroke-linecap="round"/>
  <path d="M180 135 Q165 115 160 95" stroke="#6b4a30" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M180 135 Q170 155 162 170" stroke="#6b4a30" stroke-width="4" fill="none" stroke-linecap="round"/>

  <!-- Right big -->
  <path d="M320 195 Q355 168 395 148" stroke="#5a3e28" stroke-width="9" fill="none" stroke-linecap="round"/>
  <path d="M395 148 Q425 133 460 128" stroke="#5a3e28" stroke-width="6" fill="none" stroke-linecap="round"/>
  <!-- Right sub -->
  <path d="M395 148 Q412 122 420 98" stroke="#6b4a30" stroke-width="5" fill="none" stroke-linecap="round"/>
  <path d="M460 128 Q475 108 478 88" stroke="#6b4a30" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M460 128 Q470 148 472 165" stroke="#6b4a30" stroke-width="4" fill="none" stroke-linecap="round"/>

  <!-- Center top -->
  <path d="M320 155 Q320 120 318 90" stroke="#5a3e28" stroke-width="7" fill="none" stroke-linecap="round"/>
  <path d="M318 90 Q305 68 295 52" stroke="#6b4a30" stroke-width="5" fill="none" stroke-linecap="round"/>
  <path d="M318 90 Q332 68 342 52" stroke="#6b4a30" stroke-width="5" fill="none" stroke-linecap="round"/>

  <!-- ── Foliage clusters ── -->
  <!-- Far left -->
  <circle cx="165" cy="88" r="32" fill="#2d4a2d" opacity="0.85"/>
  <circle cx="148" cy="100" r="22" fill="#3d6b3d" opacity="0.8"/>
  <circle cx="183" cy="94" r="20" fill="#3d6b3d" opacity="0.7"/>
  <circle cx="162" cy="72" r="18" fill="#4a7a4a" opacity="0.65"/>

  <!-- Left mid -->
  <circle cx="218" cy="98" r="26" fill="#2d4a2d" opacity="0.82"/>
  <circle cx="204" cy="110" r="18" fill="#3d6b3d" opacity="0.75"/>
  <circle cx="232" cy="106" r="17" fill="#4a7a4a" opacity="0.7"/>

  <!-- Left low -->
  <circle cx="163" cy="163" r="22" fill="#3d6b3d" opacity="0.75"/>
  <circle cx="150" cy="175" r="15" fill="#4a7a4a" opacity="0.65"/>

  <!-- Center top -->
  <circle cx="295" cy="46" r="24" fill="#2d4a2d" opacity="0.85"/>
  <circle cx="280" cy="58" r="16" fill="#3d6b3d" opacity="0.75"/>
  <circle cx="342" cy="46" r="24" fill="#2d4a2d" opacity="0.85"/>
  <circle cx="357" cy="58" r="16" fill="#3d6b3d" opacity="0.75"/>
  <circle cx="318" cy="30" r="20" fill="#4a7a4a" opacity="0.7"/>

  <!-- Right mid -->
  <circle cx="420" cy="90" r="26" fill="#2d4a2d" opacity="0.82"/>
  <circle cx="408" cy="103" r="18" fill="#3d6b3d" opacity="0.75"/>
  <circle cx="434" cy="100" r="17" fill="#4a7a4a" opacity="0.7"/>

  <!-- Far right -->
  <circle cx="477" cy="80" r="32" fill="#2d4a2d" opacity="0.85"/>
  <circle cx="460" cy="92" r="22" fill="#3d6b3d" opacity="0.8"/>
  <circle cx="493" cy="90" r="20" fill="#3d6b3d" opacity="0.7"/>
  <circle cx="476" cy="62" r="18" fill="#4a7a4a" opacity="0.65"/>

  <!-- Right low -->
  <circle cx="473" cy="158" r="22" fill="#3d6b3d" opacity="0.75"/>
  <circle cx="486" cy="170" r="15" fill="#4a7a4a" opacity="0.65"/>

  <!-- Highlight dots (dew) -->
  <circle cx="158" cy="82" r="3" fill="#a8d4a8" opacity="0.5"/>
  <circle cx="294" cy="40" r="2.5" fill="#a8d4a8" opacity="0.45"/>
  <circle cx="476" cy="72" r="3" fill="#a8d4a8" opacity="0.5"/>
  <circle cx="422" cy="85" r="2" fill="#a8d4a8" opacity="0.4"/>

  <!-- Gold acorns / fruits -->
  <circle cx="172" cy="165" r="5" fill="#b8922a"/>
  <circle cx="172" cy="165" r="3" fill="#d4ae55"/>
  <rect x="170" y="158" width="4" height="5" rx="1" fill="#5a3e28"/>

  <circle cx="470" cy="160" r="5" fill="#b8922a"/>
  <circle cx="470" cy="160" r="3" fill="#d4ae55"/>
  <rect x="468" y="153" width="4" height="5" rx="1" fill="#5a3e28"/>

  <circle cx="320" cy="28" r="5" fill="#b8922a"/>
  <circle cx="320" cy="28" r="3" fill="#d4ae55"/>
  <rect x="318" y="21" width="4" height="5" rx="1" fill="#5a3e28"/>

  <!-- Central shield on trunk -->
  <path d="M308 222 L332 222 L332 244 L320 252 L308 244 Z" fill="#f5edd6" stroke="#b8922a" stroke-width="1.5"/>
  <text x="320" y="241" text-anchor="middle" font-family="serif" font-size="13" fill="#b8922a">⚜</text>
</svg>
</div>
""", unsafe_allow_html=True)
def semantic_search_surname(query: str, top_k: int = 3):
    query_norm = normalize_text(query)
    query_embedding = embedding_model.encode(query_norm, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, surname_embeddings)[0]
    top_results = torch.topk(scores, k=min(top_k, len(surname_variant_rows)))

    matches = []
    seen_group_ids = set()

    for score, idx in zip(top_results.values, top_results.indices):
        row = surname_variant_rows[int(idx)]
        group = row["group_data"]
        group_id = group.get("group_id")

        if group_id in seen_group_ids:
            continue

        seen_group_ids.add(group_id)

        matches.append(
            {
                "score": round(float(score), 3),
                "group": group,
                "matched_variant": row["variant"],
            }
        )

    return matches


def search_firstname_groups(query: str, top_k: int = 3):
    query_norm = normalize_text(query)
    query_soundex = soundex(query)
    matches = []

    for group in firstname_group_summaries:
        variants = group.get("variants", [])
        normalized_variants = [normalize_text(variant) for variant in variants]

        matched_variant = None
        priority = -1

        if query_norm in normalized_variants:
            matched_variant = variants[normalized_variants.index(query_norm)]
            priority = 3
        elif any(variant_norm.startswith(query_norm) or query_norm.startswith(variant_norm) for variant_norm in normalized_variants):
            match_index = next(
                index
                for index, variant_norm in enumerate(normalized_variants)
                if variant_norm.startswith(query_norm) or query_norm.startswith(variant_norm)
            )
            matched_variant = variants[match_index]
            priority = 2
        elif query_soundex and query_soundex == group.get("soundex_code"):
            matched_variant = variants[0] if variants else group.get("display_name", "")
            priority = 1

        if priority < 0:
            continue

        matches.append(
            {
                "priority": priority,
                "group": group,
                "matched_variant": matched_variant,
                "distance": abs(len(query_norm) - len(normalize_text(matched_variant))) if matched_variant else 99,
            }
        )

    matches.sort(key=lambda item: (-item["priority"], item["distance"], item["group"].get("display_name", "")))
    return matches[:top_k]


def build_comparison_dataframe(summary_items):
    rows = []
    for item in summary_items:
        metrics = item.get("metrics", {})
        rows.append(
            {
                "Approach": item.get("approach_name", ""),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1": metrics.get("f1_score"),
                "False Merge": metrics.get("false_merge_rate"),
                "False Split": metrics.get("false_split_rate"),
                "TP": metrics.get("true_positive_pairs"),
                "FP": metrics.get("false_positive_pairs"),
                "FN": metrics.get("false_negative_pairs"),
            }
        )

    return pd.DataFrame(rows)
# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏰  Noms de Famille", "📜  Prénoms", "🗺  Visualisations"])

with tab1:
    st.markdown('<p class="tab-header">Rechercher un nom de famille</p>', unsafe_allow_html=True)
    query = st.text_input("Entrez un patronyme", placeholder="ex: Dupont, Martin, Leclerc…", label_visibility="visible")
    if query:
        matches = semantic_search_surname(query, top_k=3)

        if matches:
            for match in matches:
                item = match["group"]
                variants = item.get("variants", [])
                pills = "".join(f'<span class="variant-pill">{v}</span>' for v in variants)
                score = item.get("confidence_score", "N/A")
                summary = item.get("summary", "")
                semantic_score = match["score"]
                matched_variant = match["matched_variant"]

                st.markdown(f"""
                <div class="result-card">
                <h3>Groupe trouvé</h3>
                <div style="margin-bottom:0.6rem">{pills}</div>
                <p><strong>Variante la plus proche trouvée :</strong> {matched_variant}</p>
                <p><strong>Score sémantique :</strong> {semantic_score}</p>
                <p class="summary-text">{summary}</p>
                <span class="score-badge">Confiance : {score}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Aucun patronyme correspondant trouvé dans notre corpus.")

with tab2:
    st.markdown('<p class="tab-header">Rechercher un prénom</p>', unsafe_allow_html=True)
    query_first = st.text_input("Entrez un prénom", placeholder="ex: Jean, Marie, Louis…", label_visibility="visible")

    if query_first:
        found = False

        if firstname_group_summaries:
            matches = search_firstname_groups(query_first, top_k=3)
            for match in matches:
                found = True
                item = match["group"]
                variants = item.get("variants", [])
                pills = "".join(f'<span class="variant-pill">{v}</span>' for v in variants)
                origins = ", ".join(item.get("origins", [])) or "Non renseignee"
                meanings = ", ".join(item.get("meanings", [])[:3]) or "Non renseignee"
                score = item.get("quality_score", "N/A")
                summary = item.get("summary", "")
                label = item.get("display_name", "") or match["matched_variant"]
                matched_variant = match["matched_variant"]
                soundex_code = item.get("soundex_code", "")

                st.markdown(f"""
                <div class="result-card">
                  <h3>{label}</h3>
                  <div style="margin-bottom:0.6rem">{pills}</div>
                  <p><strong>Variante retrouvee :</strong> {matched_variant}</p>
                  <p><strong>Code Soundex :</strong> {soundex_code}</p>
                  <p><strong>Origine(s) :</strong> {origins}</p>
                  <p><strong>Signification(s) :</strong> {meanings}</p>
                  <p class="summary-text">{summary}</p>
                  <span class="score-badge">Score qualite : {score}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            for item in firstname_summaries:
                if query_first.lower() == item.get("first_name", "").lower():
                    found = True
                    score = item.get("quality_score", "N/A")
                    summary = item.get("summary", "")
                    name = item.get("first_name", "")
                    st.markdown(f"""
                    <div class="result-card">
                      <h3>{name}</h3>
                      <p class="summary-text">{summary}</p>
                      <span class="score-badge">Score qualité : {score}</span>
                    </div>
                    """, unsafe_allow_html=True)
        if not found:
            st.warning("Aucun prénom correspondant trouvé dans notre corpus.")

with tab3:
    st.markdown('<p class="tab-header">Visualisations générées</p>', unsafe_allow_html=True)

    score_file = RESULTS_DIR / "model_scores.png"
    comparison_prf_file = TEST_VISUALS_DIR / "comparison_prf.svg"

    if score_file.exists():
        st.markdown("### Comparaison des scores des modèles de résumé de texte")
        st.image(str(score_file), use_container_width=True)
    else:
        st.info("Scores non disponibles.")

    if comparison_prf_file.exists():
        st.markdown("### Evaluation des modèles de regroupement des noms de famille")
        st.image(str(comparison_prf_file), use_container_width=True)
    else:
        st.info("Le graphique d'evaluation des modeles de regroupement textuel n'est pas disponible.")
    # ── Origins visualization ─────────────────────────────

    # if firstname_dataset:
    #     st.markdown("### Répartition des origines des prénoms")

    #     origin_rows = []

    #     for item in firstname_dataset:
    #         origin = item.get("origin", "").strip()
    #         if origin:
    #             origin_rows.append({"origin": origin})

    #     if origin_rows:
    #         df = pd.DataFrame(origin_rows)

    #         origin_counts = (
    #             df.groupby("origin")
    #             .size()
    #             .reset_index(name="count")
    #             .sort_values("count", ascending=False)
    #         )

    #         fig = px.bar(
    #             origin_counts,
    #             x="origin",
    #             y="count",
    #             title="Nombre de prénoms par origine"
    #         )

    #         st.plotly_chart(fig, width="stretch")
