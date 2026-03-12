## Project Overview

Ce projet regroupe automatiquement des variantes de noms de famille a partir:

- du nom lui-meme
- du contexte textuel contenu dans la description d'origine

Le but actuel est de tester plusieurs strategies de regroupement et de comparer leurs sorties de maniere qualitative.

## Current Data Flow

Le script principal est [`code/main.py`](/d:/Nlp%20naming%20project/code/main.py).

Pipeline actuel:

1. charger [`data/names.json`](/d:/Nlp%20naming%20project/data/names.json) et [`data/origins.json`](/d:/Nlp%20naming%20project/data/origins.json)
2. construire un dataset structure avec `name`, `origin_id`, `origin_text`
3. nettoyer et normaliser les noms
4. generer des candidats proches avec des heuristiques simples
5. appliquer une strategie de similarite selon l'approche choisie
6. produire des groupes et un `final_dataset` compact

## Critere Commun De Regroupement

Toutes les approches partagent le meme mecanisme de regroupement dans [`code/main.py`](/d:/Nlp%20naming%20project/code/main.py):

1. chaque nom est d'abord normalise (`lowercase`, suppression des accents, ponctuation retiree, lemmatisation simple, puis concatenation en `normalized_name`)
2. les comparaisons se font entre variantes normalisees, pas directement entre les graphies brutes
3. un candidat n'est compare que s'il passe les memes filtres heuristiques:
   - meme premiere lettre
   - difference de longueur inferieure ou egale a 3
   - mise en panier preliminaire par prefixe de 3 caracteres + longueur
4. deux variantes sont placees dans le meme groupe si leur score de similarite est superieur ou egal au seuil de l'approche
5. la difference entre les modeles porte donc sur le calcul du score, pas sur la logique generale de formation des groupes

En pratique, le critere commun entre tous les modeles est:

`meme espace de noms normalises` + `memes filtres de candidats` + `meme regle score >= seuil`

## Implemented Approaches

### 1. Name Similarity

- dossier: [`outputs/01_name_similarity`](/d:/Nlp%20naming%20project/outputs/01_name_similarity)
- principe: embeddings sur le nom normalise uniquement
- modele: `sentence-transformers/all-MiniLM-L6-v2`

### 2. Name And Context

- dossier: [`outputs/02_name_and_context`](/d:/Nlp%20naming%20project/outputs/02_name_and_context)
- principe: embeddings sur `nom + contexte de description`
- modele: `sentence-transformers/all-MiniLM-L6-v2`

### 3. SequenceMatcher

- dossier: [`outputs/03_sequence_matcher`](/d:/Nlp%20naming%20project/outputs/03_sequence_matcher)
- principe: similarite orthographique avec `difflib.SequenceMatcher`

### 4. Levenshtein

- dossier: [`outputs/04_levenshtein`](/d:/Nlp%20naming%20project/outputs/04_levenshtein)
- principe: distance de Levenshtein transformee en score de similarite
- implementation: version Python integree au projet

### 5. Soundex

- dossier: [`outputs/05_soundex`](/d:/Nlp%20naming%20project/outputs/05_soundex)
- principe: regroupement phonétique avec un code Soundex
- implementation: version Python integree au projet

### 6. spaCy

- dossier: [`outputs/06_spacy`](/d:/Nlp%20naming%20project/outputs/06_spacy)
- principe: similarite `spaCy` sur `nom + contexte`
- modele `spaCy`: `fr_core_news_md`

## Output Files

Chaque dossier d'approche contient ses propres fichiers avec un suffixe explicite:

- `structured_dataset_<approach>.csv`
- `structured_dataset_<approach>.json`
- `cleaned_dataset_<approach>.json`
- `name_groups_<approach>.json`
- `final_dataset_<approach>.json`
- `run_metadata_<approach>.json`

Le `final_dataset` suit actuellement ce format:

```json
[
    {
        "group_id": 1,
        "variants": ["Mitailler", "Mitaillier"],
        "origin_ids": ["O39"],
        "origin_texts": ["..."]
    }
]
```

## Current Progress

Etat d'avancement actuel:

- les 6 approches sont implementees dans [`code/main.py`](/d:/Nlp%20naming%20project/code/main.py)
- les sorties ont ete generees pour les 6 approches
- les noms de fichiers de sortie sont differencies par approche
- un script local [`code/compare_metrics.py`](/d:/Nlp%20naming%20project/code/compare_metrics.py) existe si une evaluation avec gold standard est voulue plus tard
- pour le moment, le travail principal est l'exploration et l'inspection qualitative des groupes generes

## Important Output Files

- [`outputs/01_name_similarity/name_groups_name_similarity.json`](/d:/Nlp%20naming%20project/outputs/01_name_similarity/name_groups_name_similarity.json)
- [`outputs/02_name_and_context/name_groups_name_and_context.json`](/d:/Nlp%20naming%20project/outputs/02_name_and_context/name_groups_name_and_context.json)
- [`outputs/03_sequence_matcher/name_groups_sequence_matcher.json`](/d:/Nlp%20naming%20project/outputs/03_sequence_matcher/name_groups_sequence_matcher.json)
- [`outputs/04_levenshtein/name_groups_levenshtein.json`](/d:/Nlp%20naming%20project/outputs/04_levenshtein/name_groups_levenshtein.json)
- [`outputs/05_soundex/name_groups_soundex.json`](/d:/Nlp%20naming%20project/outputs/05_soundex/name_groups_soundex.json)
- [`outputs/06_spacy/name_groups_spacy.json`](/d:/Nlp%20naming%20project/outputs/06_spacy/name_groups_spacy.json)

## Execution

Depuis la racine du projet:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r .\code\requirements.txt
```

Approche 1:

```powershell
$env:NAME_GROUP_APPROACH="approach_1_name_similarity"
python .\code\main.py
```

Approche 2:

```powershell
$env:NAME_GROUP_APPROACH="approach_2_name_and_context"
python .\code\main.py
```

Approche 3:

```powershell
$env:NAME_GROUP_APPROACH="approach_3_sequence_matcher"
python .\code\main.py
```

Approche 4:

```powershell
$env:NAME_GROUP_APPROACH="approach_4_levenshtein"
python .\code\main.py
```

Approche 5:

```powershell
$env:NAME_GROUP_APPROACH="approach_5_soundex"
python .\code\main.py
```

Approche 6:

```powershell
$env:NAME_GROUP_APPROACH="approach_6_spacy"
python .\code\main.py
```

## Useful Environment Variables

- `HF_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`
- `SPACY_MODEL=fr_core_news_md`
- `NAME_GROUP_THRESHOLD=0.78`
- `NAME_CONTEXT_GROUP_THRESHOLD=0.78`
- `SEQUENCE_MATCHER_THRESHOLD=0.88`
- `LEVENSHTEIN_THRESHOLD=0.85`
- `SOUNDEX_THRESHOLD=1.0`
- `SPACY_SIMILARITY_THRESHOLD=0.75`

## Notes

- `python-Levenshtein` et `fuzzy` n'ont pas ete installes: les approches Levenshtein et Soundex sont implementees directement en Python dans le projet.
- `spaCy` necessite un pipeline installe, actuellement `fr_core_news_md`.
- les sorties generees sont destinees a l'analyse et ne devraient pas etre considerees comme une verite terrain sans validation manuelle.
