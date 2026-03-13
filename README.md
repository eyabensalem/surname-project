# Name Origins NLP Project

Projet NLP de regroupement de variantes de noms de famille et de prenoms, avec generation de resumes, evaluation, scraping et application Streamlit.

## Overview

Ce projet combine deux volets :

- `noms de famille` : regroupement automatique de variantes de patronymes a partir de `data/names.json` et `data/origins.json`
- `prenoms` : scraping de fiches, structuration des details, resume des descriptions et regroupement phonetique avec Soundex

Le pipeline complet produit des fichiers dans `outputs/` et `results/`, puis les affiche dans une application interactive Streamlit.

## Current project state

- Le pipeline principal des noms de famille utilise par defaut `approach_5_soundex`
- Les sorties principales synchronisees vers l'application sont :
  - `results/final_dataset.json`
  - `results/merged_groups.json`
  - `results/group_summaries.json`
- Les prenoms sont scrapes dans `results/firstnames_dataset.json`, puis regroupes dans :
  - `results/firstnames_grouped_soundex.json`
  - `results/firstnames_group_summaries_soundex.json`

## Repository structure

```text
app/
  streamlit_app.py              # application finale
code/
  main.py                       # pipeline principal de regroupement des noms
data/
  names.json                    # noms de famille bruts
  origins.json                  # textes d'origine associes
outputs/
  01_name_similarity/
  02_name_and_context/
  03_sequence_matcher/
  04_levenshtein/
  05_soundex/
  06_spacy/
results/
  final_dataset.json
  merged_groups.json
  group_summaries.json
  firstnames_list.json
  firstnames_dataset.json
  firstnames_summaries.json
  firstnames_grouped_soundex.json
  firstnames_group_summaries_soundex.json
src/
  config.py
  run_all.py
  compare_summarizers.py
  evaluate_summaries.py
  plot_model_scores.py
  scrape_firstname_list.py
  scrape_firstname_details.py
  summarize_firstnames.py
  group_firstnames_soundex.py
test/
  data/
    test_data.json
    gold_clusters.template.json
  run_test_approaches.py
  compare_test_metrics.py
```

## Main pipeline

### 1. Noms de famille

Le pipeline principal est dans `code/main.py`.

Il :

1. charge les noms et leurs textes d'origine
2. normalise les noms
3. applique une approche de regroupement
4. produit un `final_dataset_<approach>.json`
5. fusionne les textes du groupe
6. genere un resume de groupe

Par defaut, `src/run_all.py` lance `code/main.py` avec `approach_5_soundex`, puis copie les sorties Soundex vers `results/` pour l'application.

### 2. Prenoms

Le pipeline prenoms est dans `src/` :

1. `scrape_firstname_list.py` recupere la liste des prenoms
2. `scrape_firstname_details.py` recupere les details de chaque fiche
3. `summarize_firstnames.py` genere des resumes simples
4. `group_firstnames_soundex.py` regroupe les prenoms par Soundex et cree des resumes de groupe

### 3. Application

`app/streamlit_app.py` affiche :

- une recherche sur les groupes de noms de famille
- une recherche sur les groupes de prenoms
- des visualisations d'evaluation et de comparaison

## Shared grouping principle

Tous les modeles de regroupement suivent la meme logique metier :

1. normaliser les noms
2. comparer seulement des candidats plausibles
3. mesurer une proximite entre deux noms
4. fusionner les noms si la preuve est suffisante
5. construire un groupe unique pour chaque variante

Autrement dit, tous les modeles essaient de repondre a la meme question :

> Est-ce que ces deux formes renvoient au meme patronyme malgre des variations d'ecriture, de prononciation ou de contexte ?

Ce qui change d'un modele a l'autre, c'est le type de similarite utilise.

## Grouping models

### approach_1_name_similarity

Principe :

- transforme chaque nom normalise en embedding avec `SentenceTransformer`
- compare les vecteurs avec une similarite cosinus
- groupe les noms si leur similarite depasse un seuil

Idee :

- utile si deux noms ont une forme generale proche
- reste faible sur des variantes purement orthographiques fines

### approach_2_name_and_context

Principe :

- construit un texte `nom + descriptions`
- encode ce texte avec `SentenceTransformer`
- compare les embeddings par similarite cosinus

Idee :

- utilise a la fois la forme du nom et le contexte textuel
- peut mieux capter des noms proches si leurs descriptions racontent la meme origine

### approach_3_sequence_matcher

Principe :

- compare directement deux chaines avec `difflib.SequenceMatcher`
- calcule un ratio de ressemblance caractere par caractere

Idee :

- simple et interpretable
- bon pour des variantes orthographiques courtes

### approach_4_levenshtein

Principe :

- calcule combien d'operations d'edition il faut pour passer d'un nom a l'autre
- convertit cette distance en ratio de similarite

Idee :

- tres adapte aux fautes, ajouts, suppressions ou substitutions mineures
- plus strict qu'un modele semantique

### approach_5_soundex

Principe :

- transforme chaque nom en code phonetique Soundex
- groupe deux noms s'ils ont le meme code

Idee :

- specialement utile pour les variantes qui se prononcent de facon proche
- c'est l'approche retenue comme pipeline principal du projet

### approach_6_spacy

Principe :

- cree un document spaCy a partir du nom et du contexte
- compare les documents avec `doc.similarity`

Idee :

- cherche une proximite vectorielle plus large
- peut retrouver des liens contextuels, mais peut aussi faire plus de sur-fusions

## Output format for surname grouping

Chaque approche produit un fichier `final_dataset_<approach>.json` de cette forme :

```json
[
  {
    "group_id": 1,
    "variants": ["nom_a", "nom_b"],
    "origin_ids": ["T00001", "T00002"],
    "origin_texts": ["texte 1", "texte 2"]
  }
]
```

## Surname summary generation

En plus du regroupement, le projet genere des resumes de groupes de noms.

Dans le pipeline principal, le resume est extractif :

- decoupage en phrases
- score par mots-cles metier
- ponderation TF-IDF
- bonus de position
- selection des meilleures phrases

Les fichiers produits sont :

- `merged_groups_<approach>.json`
- `group_summaries_<approach>.json`

## Summarization models compared in src/

Le projet compare aussi trois modeles de resume sur les groupes fusionnes :

### TF-IDF + keyword scoring

- resume extractif
- selectionne les phrases les plus informatives selon TF-IDF et des mots-cles metier

### TextRank

- resume extractif base sur un graphe de phrases
- garde les phrases centrales du texte

### DistilBART

- resume abstractive via un modele transformer pre-entraine
- utilise `sshleifer/distilbart-cnn-12-6`

## Evaluation

### Evaluation des modeles de regroupement

Les approches de regroupement sont comparees dans `test/` :

- `test/run_test_approaches.py` genere les sorties sur `test/data/test_data.json`
- `test/compare_test_metrics.py` compare les predictions au gold de `test/data/gold_clusters.template.json`

Les metriques calculees sont :

- precision
- recall
- F1
- false merge rate
- false split rate
- TP / FP / FN

Cette evaluation est `pairwise` :

- on compare les paires de noms qui devraient etre dans le meme groupe
- pas seulement les groupes visuellement

### Evaluation des modeles de resume

`src/evaluate_summaries.py` evalue les resumes avec :

- `ROUGE-1`
- `ROUGE-L`

sur quelques resumes de reference manuels.

## Installation

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the project

### Pipeline complet

```powershell
.\venv\Scripts\python.exe src\run_all.py
```

### Application Streamlit

```powershell
.\venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

### Benchmark des modeles de regroupement

```powershell
.\venv\Scripts\python.exe test\run_test_approaches.py
.\venv\Scripts\python.exe test\compare_test_metrics.py
```

## Important notes

- `src/run_all.py` lance d'abord le pipeline principal des noms de famille avec Soundex
- l'application lit ensuite les fichiers de `results/`
- pour les prenoms, il faut regenerer les sorties derivees apres un nouveau scraping si `firstnames_dataset.json` change
- les performances des modeles dependent fortement de la qualite du gold de reference

## Main files to mention in a report

- `code/main.py` : coeur du regroupement des noms de famille
- `src/run_all.py` : orchestration du pipeline final
- `app/streamlit_app.py` : interface utilisateur
- `test/run_test_approaches.py` : benchmark des approches de regroupement
- `test/compare_test_metrics.py` : evaluation pairwise des regroupements
