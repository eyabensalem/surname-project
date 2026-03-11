# Surname Project

## Project overview
This project analyzes surname variants and their origins using NLP techniques.

The work is divided into two main parts:
- **Yasmine**: detection and grouping of surname variants
- **Eya**: text merging, summarization, and first-name scraping extension

## Current pipeline
1. Load grouped surname variants
2. Retrieve matching origin texts
3. Merge texts for each group
4. Generate automatic summaries

## Project structure

```text
surname-project/
│
├── data/
│   └── origins.json
│
├── results/
│   ├── grouped_names.json
│   ├── merged_groups.json
│   └── group_summaries.json
│
├── src/
│   ├── text_grouping.py
│   └── summarization.py
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore

How to run
1. Create and activate a virtual environment
python -m venv venv
2. Install dependencies
pip install -r requirements.txt
3. Run text merging
python src/text_grouping.py
4. Run summarization
python src/summarization.py
Output files

results/merged_groups.json: merged origin texts by surname group

results/group_summaries.json: generated summaries for each group


---

## 3) Fais les commandes

```powershell
git add .
git commit -m "Add project documentation and dependencies"
git push