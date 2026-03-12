## Project Overview

Ce projet vise a regrouper automatiquement des variantes de noms de famille a partir de:

- la forme du nom
- les descriptions d'origine associees

L'objectif actuel est de produire des groupes de variantes comparables entre plusieurs approches de regroupement, puis d'analyser qualitativement leurs differences.

## Current Pipeline

Le script principal est [`code/main.py`](/d:/Nlp%20naming%20project/code/main.py).

Le pipeline actuel est:

1. charger les donnees source depuis [`data/names.json`](/d:/Nlp%20naming%20project/data/names.json) et [`data/origins.json`](/d:/Nlp%20naming%20project/data/origins.json)
2. construire un dataset structure `name / origin_id / origin_text`
3. nettoyer les noms:
   - minuscule
   - suppression des accents
   - suppression de la ponctuation
   - tokenisation
   - lemmatisation simple
4. encoder les noms avec `sentence-transformers/all-MiniLM-L6-v2`
5. comparer les candidats avec une similarite cosinus
6. produire des groupes et un `final_dataset` compact

Le code limite aussi les comparaisons avec des heuristiques simples:

- meme debut de mot
- longueurs proches
- meme premiere lettre

## Available Approaches

### 1. Name Similarity

Dossier de sortie:

- [`outputs/01_name_similarity`](/d:/Nlp%20naming%20project/outputs/01_name_similarity)

Principe:

- comparaison basee sur le nom normalise uniquement

Fichiers generes:

- `structured_dataset_name_similarity.csv`
- `structured_dataset_name_similarity.json`
- `cleaned_dataset_name_similarity.json`
- `name_groups_name_similarity.json`
- `final_dataset_name_similarity.json`
- `run_metadata_name_similarity.json`

### 2. Name And Context

Dossier de sortie:

- [`outputs/02_name_and_context`](/d:/Nlp%20naming%20project/outputs/02_name_and_context)

Principe:

- comparaison basee sur `nom + contexte de description`
- meme modele d'embeddings que l'approche 1

Fichiers generes:

- `structured_dataset_name_and_context.csv`
- `structured_dataset_name_and_context.json`
- `cleaned_dataset_name_and_context.json`
- `name_groups_name_and_context.json`
- `final_dataset_name_and_context.json`
- `run_metadata_name_and_context.json`

## Output Format

Le `final_dataset` est actuellement en format compact, par groupe:

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

Ce format est destine a faciliter:

- la lecture des groupes
- la comparaison entre approches
- l'inspection manuelle des variantes

## Current Progress

Etat actuel du projet:

- les deux approches sont implementees dans [`code/main.py`](/d:/Nlp%20naming%20project/code/main.py)
- les sorties des deux approches sont deja generees dans leurs dossiers respectifs
- les noms de fichiers de sortie ont ete rendus explicites pour eviter toute confusion entre approches
- un script local [`code/compare_metrics.py`](/d:/Nlp%20naming%20project/code/compare_metrics.py) existe pour calculer des metriques si une verite terrain est ajoutee plus tard
- pour le moment, la comparaison finale des approches n'est pas encore lancee avec un vrai fichier `gold`

## Useful Files

- [`outputs/01_name_similarity/name_groups_name_similarity.json`](/d:/Nlp%20naming%20project/outputs/01_name_similarity/name_groups_name_similarity.json)
- [`outputs/01_name_similarity/final_dataset1.json`](/d:/Nlp%20naming%20project/outputs/01_name_similarity/final_dataset1.json)
- [`outputs/02_name_and_context/name_groups_name_and_context.json`](/d:/Nlp%20naming%20project/outputs/02_name_and_context/name_groups_name_and_context.json)
- [`outputs/02_name_and_context/final_dataset2.json`](/d:/Nlp%20naming%20project/outputs/02_name_and_context/final_dataset2.json)

## Execution

Depuis la racine du projet:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r .\code\requirements.txt
```

Pour l'approche 1:

```powershell
$env:NAME_GROUP_APPROACH="approach_1_name_similarity"
python .\code\main.py
```

Pour l'approche 2:

```powershell
$env:NAME_GROUP_APPROACH="approach_2_name_and_context"
python .\code\main.py
```

## Next Logical Step

La prochaine etape logique est de comparer qualitativement les groupes produits par les deux approches, puis de decider si une evaluation avec verite terrain doit etre ajoutee.
