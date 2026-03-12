## Experiment Outputs

- `01_name_similarity`: regroupement par similarite entre noms.
- `02_name_and_context`: regroupement par similarite entre noms et descriptions avec le meme modele d'embeddings.

Chaque dossier contient ses propres sorties:

- `structured_dataset_name_similarity.csv` ou `structured_dataset_name_and_context.csv`
- `structured_dataset_name_similarity.json` ou `structured_dataset_name_and_context.json`
- `cleaned_dataset_name_similarity.json` ou `cleaned_dataset_name_and_context.json`
- `name_groups_name_similarity.json` ou `name_groups_name_and_context.json`
- `final_dataset_name_similarity.json` ou `final_dataset_name_and_context.json`
- `run_metadata_name_similarity.json` ou `run_metadata_name_and_context.json`

Variables d'environnement utiles:

- `NAME_GROUP_APPROACH=approach_1_name_similarity`
- `NAME_GROUP_APPROACH=approach_2_name_and_context`
- `NAME_GROUP_THRESHOLD=0.78`
- `NAME_CONTEXT_GROUP_THRESHOLD=0.78`
- `HF_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`

Notes:

- L'approche 2 encode `nom + contexte de description` avec le meme modele que l'approche 1.

## Execution

Depuis la racine du projet:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r .\code\requirements.txt
```

```powershell
$env:NAME_GROUP_APPROACH="approach_1_name_similarity"
python .\code\main.py
```

```powershell
$env:NAME_GROUP_APPROACH="approach_2_name_and_context"
python .\code\main.py
```
