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

```

## 📝 Génération automatique de résumés

Après le regroupement des variantes de noms, plusieurs textes d’origine peuvent être associés à un même groupe.

Ces textes peuvent être :
- longs
- redondants
- issus de sources différentes

L’objectif est donc de générer un **résumé court et représentatif** pour chaque groupe.

### Pipeline de résumé

```

Textes d'origine
↓
Fusion des textes
↓
Extraction des phrases importantes
↓
Résumé automatique

```

---

## 🔬 Modèles de résumé testés

Trois approches ont été comparées pour générer les résumés.

### 1. TF-IDF + mots clés

Type : **résumé extractif**

Principe :

1. découper le texte en phrases
2. calculer l’importance des mots avec **TF-IDF**
3. ajouter un score basé sur des mots clés métier

Exemples de mots clés :

```

origine
signifie
variante
désigne
forme

```

Score d’une phrase :

```

score = score TF-IDF + score mots clés

```

Les **2 phrases les mieux classées** sont sélectionnées pour construire le résumé.

Avantages :

- rapide
- interprétable
- adapté à des textes courts et factuels

---

### 2. TextRank

Type : **résumé extractif**

TextRank est une méthode inspirée de l’algorithme **PageRank**.

Principe :

1. chaque phrase devient un **nœud d’un graphe**
2. la similarité entre phrases est calculée
3. les phrases les plus **centrales dans le graphe** sont sélectionnées

Avantages :

- méthode classique de summarization
- ne nécessite pas d'entraînement

---

### 3. DistilBART

Type : **résumé abstractive**

Contrairement aux méthodes extractives, ce modèle **génère un nouveau texte** au lieu de sélectionner des phrases.

Modèle utilisé :

```

sshleifer/distilbart-cnn-12-6

```

Fonctionnement :

```

texte long
↓
transformer encoder
↓
transformer decoder
↓
résumé généré

```

Avantages :

- peut reformuler le texte
- résumé plus naturel

Limites :

- parfois moins fidèle au texte original
- modèle plus lourd

---

## 📊 Évaluation des résumés

Les résumés générés sont évalués avec la métrique **ROUGE**.

Le principe consiste à comparer :

```

résumé généré
vs
résumé de référence

```

Trois métriques principales sont utilisées :

| métrique | description |
|--------|-------------|
| ROUGE-1 | chevauchement des mots |
| ROUGE-2 | chevauchement des paires de mots |
| ROUGE-L | plus longue sous-séquence commune |

Les scores calculés :

```

ROUGE-1 F1
ROUGE-L F1

```

Un score élevé signifie que le résumé :

- contient les informations importantes
- reste proche du résumé de référence

---

## ⚙ Implémentation

L’évaluation est implémentée dans :

```

evaluate_summaries.py

```

Librairie utilisée :

```

rouge_score

```

Pipeline :

```

résumé généré
↓
comparaison avec résumé de référence
↓
calcul des scores ROUGE
↓
sauvegarde dans evaluation_results.json

```

---

## 🏆 Modèle retenu

Après comparaison, la méthode **TF-IDF + mots clés** donne les meilleurs résultats.

| modèle | résultat |
|------|--------|
TF-IDF + keywords | ⭐ meilleur |
TextRank | correct |
DistilBART | moins précis |

Cela s’explique par la nature des textes d’origine :

- courts
- factuels
- structurés

Dans ce contexte, une **méthode extractive simple est plus efficace qu’un modèle génératif**.
```

---

# Maintenant la partie scraping à mettre dans README

Ajoute une section :

```markdown
## 🌐 Web scraping des prénoms

Les informations sur les prénoms sont récupérées automatiquement depuis le site **OrigineNom**.

Le scraping est réalisé en deux étapes.

### 1. Récupération de la liste des prénoms

Script :

```

scrape_firstname_list.py

```

Ce script parcourt les pages :

```

[https://originenom.com/liste-des-prenoms/](https://originenom.com/liste-des-prenoms/)

```

et extrait :

- le prénom
- l’URL de la page de détail

Exemple :

```

{
"first_name": "Abel",
"url": "[https://originenom.com/origine-du-prenom/abel/](https://originenom.com/origine-du-prenom/abel/)"
}

```

Les résultats sont stockés dans :

```

firstnames_list.json

```

---

### 2. Extraction des détails

Script :

```

scrape_firstname_details.py

```

Pour chaque prénom de la liste :

1. la page est téléchargée avec **Requests**
2. le HTML est analysé avec **BeautifulSoup**
3. les informations suivantes sont extraites :

- origine
- signification
- description

Les données sont enregistrées dans :

```

firstnames_dataset.json

```

---

### Dataset final

Chaque prénom contient :

```

{
"first_name": "Abel",
"origin": "Hébraïque",
"meaning": "souffle ou vapeur",
"description": "...",
"quality_score": 3
}

```

Ces données sont ensuite utilisées pour :

- la génération de résumés
- l’exploration dans l’application Streamlit
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


<div align="center">

<br/>

**Projet NLP · Noms & Prénoms** · Soundex · TextRank · DistilBART · ROUGE

<br/>

*Projet académique — pipeline de regroupement et résumé automatique de noms propres.*

</div>