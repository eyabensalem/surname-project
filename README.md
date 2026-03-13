<div align="center">

<br/>

# 🔤 NLP · Noms & Prénoms

### Regroupement, résumé automatique et exploration de noms propres par similarité phonétique et sémantique.

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io)
[![Soundex](https://img.shields.io/badge/Phonétique-Soundex-8B5CF6?style=for-the-badge)](https://en.wikipedia.org/wiki/Soundex)
[![ROUGE](https://img.shields.io/badge/Évaluation-ROUGE-F59E0B?style=for-the-badge)](https://en.wikipedia.org/wiki/ROUGE_(metric))

<br/>

> Partir de noms bruts → regrouper les variantes → fusionner les textes → générer des résumés → tout explorer dans une interface.

<br/>

</div>

---

## 🗺️ Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Structure du projet](#-structure-du-projet)
- [Pipeline — Noms de famille](#-pipeline--noms-de-famille)
- [Pipeline — Prénoms](#-pipeline--prénoms)
- [Modèles de regroupement](#-modèles-de-regroupement)
- [Génération de résumés](#-génération-de-résumés)
- [Évaluation](#-évaluation)
- [Fichiers clés](#-fichiers-clés)
- [Commandes](#-commandes)
- [Résumé pour l'oral](#-résumé-pour-loral)

---

## 🎯 Vue d'ensemble

Ce projet NLP construit un pipeline complet autour des noms propres :

| Étape | Description |
|---|---|
| 🧹 **Normalisation** | Nettoyage et standardisation des noms bruts |
| 🔗 **Regroupement** | Détection automatique des variantes d'un même nom |
| 📝 **Fusion** | Agrégation des textes d'origine par groupe |
| 🤖 **Résumé** | Génération automatique d'un résumé par groupe |
| 🔍 **Exploration** | Interface Streamlit pour rechercher et visualiser |

Le modèle principal retenu pour les noms de famille est **Soundex** (`approach_5_soundex`).

---

## 📁 Structure du projet

```
projet-nlp/
│
├── data/                          # Données brutes d'entrée
│   ├── names.json                 # Liste des noms de famille
│   └── origins.json               # Textes d'origine associés
│
├── code/                          # ⭐ Cœur du pipeline noms de famille
│   └── main.py                    # Nettoyage → regroupement → résumés
│
├── src/                           # Scripts complémentaires
│   ├── run_all.py                 # Lance le pipeline global
│   ├── scrape_firstname_list.py   # Scrape la liste des prénoms
│   ├── scrape_firstname_details.py# Scrape les détails par prénom
│   ├── summarize_firstnames.py    # Résumés des prénoms
│   ├── group_firstnames_soundex.py# Regroupement phonétique des prénoms
│   ├── compare_summarizers.py     # Comparaison des modèles de résumé
│   └── evaluate_summaries.py      # Évaluation ROUGE des résumés
│
├── outputs/                       # Sorties détaillées par modèle
│   └── 05_soundex/
│       ├── final_dataset_soundex.json
│       ├── merged_groups_soundex.json
│       └── group_summaries_soundex.json
│
├── results/                       # ✅ Fichiers finaux lus par l'app
│   ├── final_dataset.json
│   ├── merged_groups.json
│   ├── group_summaries.json
│   ├── firstnames_dataset.json
│   └── firstnames_group_summaries_soundex.json
│
├── app/                           # Interface utilisateur
│   └── streamlit_app.py           # Application Streamlit finale
│
└── test/                          # Comparaison et évaluation des modèles
    ├── run_test_approaches.py
    ├── compare_test_metrics.py
    └── data/
        ├── test_data.json
        └── gold_clusters.template.json
```

---

## 🔄 Pipeline — Noms de famille

Le pipeline principal transforme des noms bruts en groupes enrichis avec résumés.

```
┌─────────────────────────────────┐
│  data/names.json                │
│  data/origins.json              │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  code/main.py                   │
│                                 │
│  ① Chargement des données       │
│  ② Normalisation des noms       │
│  ③ Comparaison par paires       │
│  ④ Création des groupes         │
│  ⑤ Fusion des textes            │
│  ⑥ Génération des résumés       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  outputs/05_soundex/            │
│  ├── final_dataset_soundex.json │
│  ├── merged_groups_soundex.json │
│  └── group_summaries_soundex.json│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  results/  (fichiers finaux)    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  app/streamlit_app.py  🖥️       │
└─────────────────────────────────┘
```

---

## 🔄 Pipeline — Prénoms

Les prénoms suivent une logique de scraping puis de regroupement phonétique.

```
scrape_firstname_list.py
        │
        ▼
firstnames_list.json
        │
        ▼
scrape_firstname_details.py
        │
        ▼
firstnames_dataset.json
        │
        ▼
summarize_firstnames.py
        │
        ▼
group_firstnames_soundex.py
        │
        ▼
firstnames_grouped_soundex.json
        │
        ▼
firstnames_group_summaries_soundex.json
        │
        ▼
app/streamlit_app.py  🖥️
```

---

## 🧪 Modèles de regroupement

Six approches ont été comparées pour regrouper les variantes de noms :

| # | Modèle | Principe | Signal utilisé |
|---|---|---|---|
| 1 | **Name Similarity** | Embeddings sémantiques | Forme du nom |
| 2 | **Name + Context** | Embeddings nom + texte | Nom et description |
| 3 | **Sequence Matcher** | Comparaison caractère par caractère | Chaîne brute |
| 4 | **Levenshtein** | Distance d'édition | Nombre de modifications |
| 5 | **Soundex** ⭐ | Similarité phonétique | Prononciation approximative |
| 6 | **spaCy** | Similarité vectorielle | Nom + contexte |

> **Modèle retenu : Soundex** — deux noms phonétiquement proches sont regroupés ensemble, ce qui est pertinent pour les variantes régionales et historiques de noms de famille.

---

## 📝 Génération de résumés

### Noms de famille

Le résumé est construit à partir des **textes fusionnés** du groupe. Les phrases les plus représentatives sont sélectionnées automatiquement.

### Prénoms

Le résumé est généré à partir des informations scrapées : **origine**, **signification**, **description**.

### Comparaison de modèles (`compare_summarizers.py`)

Trois approches ont été testées :

| Modèle | Type | Caractéristiques |
|---|---|---|
| **TF-IDF + mots-clés** | Extractif | Rapide, basé sur la fréquence |
| **TextRank** | Extractif | Graphe de similarité entre phrases |
| **DistilBART** | Abstractif | Génératif, reformule le contenu |

---

## 📊 Évaluation

### Regroupement — `test/`

Les groupes prédits sont comparés à un fichier de référence (`gold_clusters.template.json`) :

| Métrique | Description |
|---|---|
| **Precision** | Proportion de paires correctement regroupées |
| **Recall** | Proportion de vraies variantes retrouvées |
| **F1-score** | Moyenne harmonique précision / rappel |
| **False Merge** | Noms distincts mis dans le même groupe |
| **False Split** | Variantes du même nom dans des groupes séparés |

### Résumés — `evaluate_summaries.py`

Évaluation automatique des résumés avec **ROUGE** :

```
ROUGE-1  →  chevauchement des unigrammes
ROUGE-2  →  chevauchement des bigrammes
ROUGE-L  →  plus longue sous-séquence commune
```

---

## 🗂️ Fichiers clés

Si vous devez présenter le projet rapidement, voici les fichiers essentiels :

| Fichier | Rôle |
|---|---|
| `code/main.py` | ⭐ Cœur du regroupement des noms de famille |
| `src/run_all.py` | Lance le pipeline complet |
| `src/group_firstnames_soundex.py` | Regroupement phonétique des prénoms |
| `app/streamlit_app.py` | Interface finale de présentation |
| `test/run_test_approaches.py` | Exécution des 6 modèles de test |
| `test/compare_test_metrics.py` | Comparaison des métriques |

---

## ⚡ Commandes

### Installation

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Lancer le pipeline complet

```powershell
.\venv\Scripts\python.exe src\run_all.py
```

### Lancer l'application Streamlit

```powershell
.\venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

### Lancer les tests de comparaison

```powershell
# Exécuter les 6 approches de regroupement
.\venv\Scripts\python.exe test\run_test_approaches.py

# Comparer les métriques entre modèles
.\venv\Scripts\python.exe test\compare_test_metrics.py
```

---

## 🎤 Résumé pour l'oral

> Le projet est organisé en plusieurs modules.
>
> **`data/`** contient les données brutes — noms et textes d'origine.
>
> **`code/`** contient le pipeline principal : nettoyage, regroupement par Soundex, fusion des textes et génération de résumés.
>
> **`src/`** contient les scripts complémentaires : scraping des prénoms, résumés automatiques, évaluation ROUGE.
>
> Les fichiers générés sont stockés dans **`outputs/`** (par modèle) et **`results/`** (version finale).
>
> Enfin, **`app/streamlit_app.py`** affiche tout dans une interface interactive.

---

<div align="center">

<br/>

**Projet NLP · Noms & Prénoms** · Soundex · TextRank · DistilBART · ROUGE

<br/>

*Projet académique — pipeline de regroupement et résumé automatique de noms propres.*

</div>