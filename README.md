# Surname Project

## Project Overview

This project analyzes surname variants and their origins using Natural Language Processing (NLP).

The objective is to group surname variants, merge their origin descriptions, and generate automatic summaries.

The project is divided into two main contributions:

- **Yasmine**: detection and grouping of surname variants
- **Eya**: text merging, automatic summarization, model comparison, and first-name scraping extension

---

# Project Pipeline

The processing pipeline follows these steps:

1. Load surname variants grouped by similarity
2. Retrieve corresponding origin texts
3. Merge texts for each group
4. Generate summaries using several NLP approaches
5. Compare summarization models
6. Extend the pipeline to scraped first-name data

---

# NLP Methods Used

Three summarization approaches were tested:

### 1. TF-IDF Extractive Summarization
- Sentence scoring based on TF-IDF
- Additional weighting using domain keywords

Advantages:
- Fast
- Interpretable
- Lightweight

---

### 2. TextRank Summarization
Graph-based ranking algorithm for sentence extraction.

Advantages:
- Classical NLP approach
- Does not require training data

---

### 3. Transformer Model (BART / DistilBART)

A pretrained Transformer model used for abstractive summarization.

Advantages:
- Produces more natural summaries
- Can rephrase text instead of simply extracting sentences

---

# First-Name Scraping Extension

To demonstrate that the pipeline works on external data, we extended the project using web scraping.

Data was collected from:

https://originenom.com

The scraping process includes:

1. Extracting a list of first names
2. Visiting each first-name page
3. Extracting:
   - origin
   - meaning
   - description
4. Applying the same summarization pipeline

---

# Project Structure

```

surname-project/
│
├── data/
│   └── origins.json
│
├── results/
│   ├── grouped_names.json
│   ├── merged_groups.json
│   ├── group_summaries.json
│   ├── firstnames_list.json
│   ├── firstnames_dataset.json
│   └── model_comparison.json
│
├── src/
│   ├── text_grouping.py
│   ├── summarization.py
│   ├── scrape_firstname_list.py
│   ├── scrape_firstname_details.py
│   └── compare_summarizers.py
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore

````

---

# Installation

### 1 Create a virtual environment

```bash
python -m venv venv
````

### 2 Activate the environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 3 Install dependencies

```bash
pip install -r requirements.txt
```

---

# How to Run

### 1 Merge origin texts

```bash
python src/text_grouping.py
```

### 2 Generate summaries

```bash
python src/summarization.py
```

### 3 Scrape first-name list

```bash
python src/scrape_firstname_list.py
```

### 4 Scrape first-name details

```bash
python src/scrape_firstname_details.py
```

### 5 Compare summarization models

```bash
python src/compare_summarizers.py
```
## Model comparison

Three summarization approaches were compared:

- **TF-IDF + keyword scoring**: the main method used in the final pipeline
- **TextRank**: a classical graph-based extractive summarization method
- **BART / DistilBART**: a Transformer-based abstractive summarization model

### Observations
- TF-IDF produced the most stable and interpretable results on surname origin texts.
- TextRank provided relevant summaries but was sometimes less focused.
- BART generated more natural text in some cases, but could truncate or distort specialized information on short inputs.

For this reason, the final pipeline uses **TF-IDF + keyword scoring** as the main summarization approach.
---

# Output Files

| File                      | Description                             |
| ------------------------- | --------------------------------------- |
| `merged_groups.json`      | merged origin texts                     |
| `group_summaries.json`    | summaries generated with TF-IDF         |
| `firstnames_list.json`    | scraped list of first names             |
| `firstnames_dataset.json` | structured scraped dataset              |
| `model_comparison.json`   | comparison between summarization models |

---

# Conclusion

This project demonstrates how NLP techniques can be used to analyze surname origins and automatically generate summaries.

The approach was extended to external web data to demonstrate the robustness and generalization of the pipeline.

````

---
