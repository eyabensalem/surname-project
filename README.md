# 🧬 Name Origins NLP Project

### Analyse automatique des origines de noms de famille et prénoms avec NLP, scraping et recherche sémantique

## 📌 Aperçu du projet

**Name Origins NLP Project** est un projet Python de bout en bout qui explore les **origines, significations, variantes et résumés** de noms de famille et prénoms à l’aide de techniques de **Natural Language Processing (NLP)**.

Le projet combine :

- le **regroupement et la fusion de textes**
- la **génération automatique de résumés**
- la **comparaison de modèles de résumé**
- l’**évaluation avec ROUGE**
- le **web scraping pour les prénoms**
- la **visualisation de données**
- la **recherche sémantique avec embeddings**
- une **application interactive Streamlit**

L’objectif est de construire un pipeline complet, allant des données textuelles brutes jusqu’à une interface utilisateur interactive.

---

# 🎯 Objectifs du projet

Ce projet vise à :

- regrouper automatiquement les variantes de noms de famille
- fusionner les textes d’origine associés à un même groupe
- générer des résumés courts sur l’origine des patronymes
- comparer plusieurs approches de summarization
- évaluer la qualité des résumés avec des métriques NLP
- scraper et structurer des données sur les prénoms
- visualiser les relations entre variantes et les scores des modèles
- construire une application interactive de recherche

---

# 🧠 Fonctionnalités principales

## 1. Traitement des noms de famille
- regroupement de variantes de patronymes
- fusion des textes associés à un même groupe
- génération de résumés avec une approche extractive hybride
- calcul d’un score de confiance pour les résumés

## 2. Comparaison de méthodes de résumé
- comparaison de plusieurs approches :
  - **TF-IDF + mots-clés métier**
  - **TextRank**
  - **DistilBART**
- évaluation des sorties avec **ROUGE-1** et **ROUGE-L**

## 3. Extension prénoms
- scraping des listes de prénoms
- visite des pages détail
- extraction de :
  - l’origine
  - la signification
  - la description
- calcul d’un score qualité
- génération de résumés courts pour les prénoms

## 4. Application interactive
- recherche de noms de famille via **similarité sémantique**
- recherche directe de prénoms
- affichage des résumés générés
- visualisation des scores et graphes générés

---

# 🧩 Structure du projet

```bash
surname-project/
│
├── data/
│   └── origins.json
│
├── results/
│   ├── final_dataset.json
│   ├── merged_groups.json
│   ├── group_summaries.json
│   ├── model_comparison.json
│   ├── evaluation_results.json
│   ├── firstnames_list.json
│   ├── firstnames_dataset.json
│   ├── firstnames_summaries.json
│   ├── surname_variant_graph.png
│   └── model_scores.png
│
├── src/
│   ├── config.py
│   ├── text_grouping.py
│   ├── summarization.py
│   ├── compare_summarizers.py
│   ├── evaluate_summaries.py
│   ├── visualize_variants.py
│   ├── plot_model_scores.py
│   ├── scrape_firstname_list.py
│   ├── scrape_firstname_details.py
│   ├── summarize_firstnames.py
│   └── run_all.py
│
├── app/
│   └── streamlit_app.py
│
├── screenshots/
│   ├── page1.png
│   ├── page2.png
│   ├── page3.png
│   ├── page4.png
│   └── page5.png
│
├── requirements.txt
└── README.md
````

---

# ⚙️ Installation

## 1. Cloner le dépôt

```bash
git clone https://github.com/eyabensalem/surname-project.git
cd surname-project
```

## 2. Créer un environnement virtuel

```bash
python -m venv venv
```

## 3. Activer l’environnement

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# 🚀 Exécuter le pipeline complet

Pour lancer tout le pipeline NLP :

```bash
python src/run_all.py
```

Ce script exécute automatiquement les étapes suivantes :

1. fusion des textes par groupe de noms
2. génération des résumés de patronymes
3. comparaison des méthodes de summarization
4. évaluation avec ROUGE
5. génération des visualisations
6. scraping de la liste des prénoms
7. scraping des fiches prénom
8. génération des résumés de prénoms

Tous les fichiers générés sont enregistrés dans :

```bash
results/
```

---

# 🔄 Vue d’ensemble du pipeline

```text
Données brutes sur les noms
        ↓
Regroupement / dataset des variantes
        ↓
Fusion des textes par groupe
        ↓
Génération de résumés de noms
        ↓
Comparaison des méthodes de résumé
        ↓
Évaluation avec ROUGE
        ↓
Création des visualisations
        ↓
Scraping des prénoms
        ↓
Résumé des descriptions de prénoms
        ↓
Affichage dans l’application Streamlit
```

---

# 📂 Fichiers d’entrée et de sortie

## Entrées principales

* `data/origins.json` → données textuelles sources sur l’origine des noms
* dataset groupé des patronymes → utilisé pour relier variantes et identifiants de textes

## Sorties principales

* `results/merged_groups.json` → texte fusionné par groupe de patronymes
* `results/group_summaries.json` → résumés générés pour les noms
* `results/model_comparison.json` → comparaison des méthodes de résumé
* `results/evaluation_results.json` → résultats de l’évaluation ROUGE
* `results/firstnames_list.json` → liste des prénoms scrapés
* `results/firstnames_dataset.json` → fiches prénom structurées
* `results/firstnames_summaries.json` → résumés générés pour les prénoms
* `results/surname_variant_graph.png` → graphe des variantes de noms
* `results/model_scores.png` → graphique de comparaison des scores

---

# 🧪 Description des scripts

## `config.py`

Centralise tous les chemins de fichiers utilisés dans le projet.

## `text_grouping.py`

Charge les textes d’origine et les groupes de patronymes, puis fusionne tous les textes liés à un même groupe.

## `summarization.py`

Génère des résumés de noms de famille avec une approche **extractive hybride** fondée sur :

* le découpage en phrases
* le scoring par mots-clés métier
* la pondération TF-IDF
* des heuristiques de position et de longueur

## `compare_summarizers.py`

Compare trois approches de résumé :

* méthode personnalisée TF-IDF + mots-clés
* TextRank
* DistilBART

## `evaluate_summaries.py`

Évalue les résumés générés à l’aide de résumés de référence manuels et des métriques :

* **ROUGE-1**
* **ROUGE-L**

## `plot_model_scores.py`

Génère un graphique comparatif des performances moyennes des modèles.

## `visualize_variants.py`

Crée un graphe des relations entre variantes de patronymes avec **NetworkX**.

## `scrape_firstname_list.py`

Scrape plusieurs pages de liste de prénoms et extrait les noms et les URLs des fiches.

## `scrape_firstname_details.py`

Visite chaque fiche prénom et extrait :

* l’origine
* la signification
* la description
* un score qualité

Ce script inclut aussi :

* des règles de fallback
* du nettoyage de texte
* un filtrage des fiches trop faibles ou non pertinentes

## `summarize_firstnames.py`

Génère des résumés courts à partir des descriptions de prénoms.

## `run_all.py`

Orchestre l’ensemble du pipeline, de la préparation des données jusqu’à la génération des résultats.

---

# 🤖 Méthodes NLP utilisées

## 1. Résumé extractif hybride

Pour les patronymes, le projet utilise une méthode de summarization personnalisée qui combine :

* le découpage en phrases
* des mots-clés métier
* le **TF-IDF**
* des heuristiques sur la position et la longueur des phrases

Cette approche est légère, interprétable et adaptée au type de textes utilisés.

## 2. TextRank

Un algorithme de résumé extractif basé sur un graphe, utilisé comme méthode de comparaison.

## 3. DistilBART

Un modèle de résumé de type transformer issu de Hugging Face, utilisé pour comparer approche classique et approche neuronale.

## 4. Recherche sémantique

L’application Streamlit utilise :

* **SentenceTransformers**
* le modèle `all-MiniLM-L6-v2`
* la **similarité cosinus**

pour retrouver les variantes de noms les plus proches à partir de la saisie utilisateur.

---

# 📊 Évaluation

La qualité des résumés est évaluée avec les métriques **ROUGE**.

Métriques utilisées :

* **ROUGE-1 F1**
* **ROUGE-L F1**

Un petit ensemble de résumés de référence rédigés manuellement permet de comparer :

* la méthode TF-IDF
* TextRank
* DistilBART

Cette évaluation fournit un premier benchmark de qualité.

---

# 📈 Visualisations

## Graphe des variantes de noms

Ce graphe montre les relations entre les différentes variantes d’un même patronyme.

![Graphe des variantes](results/surname_variant_graph.png)

## Comparaison des scores des modèles

Ce graphique présente le score moyen ROUGE-1 F1 pour chaque méthode de summarization.

![Scores des modèles](results/model_scores.png)

---

# 🔎 Exemple d’extraction de prénom

Exemple de fiche prénom structurée :

```json
{
  "first_name": "Abel",
  "url": "https://originenom.com/origine-du-prenom/abel/",
  "source": "OrigineNom",
  "origin": "Hébraïque",
  "meaning": "souffle ou vapeur",
  "description": "Le prénom Abel trouve ses racines dans la Bible...",
  "quality_score": 3
}
```

Exemple de résumé généré :

```json
{
  "first_name": "Abel",
  "origin": "Hébraïque",
  "meaning": "souffle ou vapeur",
  "summary": "Le prénom Abel trouve ses racines dans la Bible. Il possède une origine hébraïque.",
  "quality_score": 3
}
```

---

# 🖥 Application Streamlit

Le projet inclut une application Streamlit qui permet :

* de rechercher un **nom de famille**
* de rechercher un **prénom**
* d’afficher les variantes d’un nom
* de consulter les résumés générés
* de voir le score sémantique
* de visualiser les scores de confiance / qualité
* d’explorer les visualisations générées

## Lancer l’application

```bash
streamlit run app/streamlit_app.py
```

Puis ouvrir :

```bash
http://localhost:8501
```

---

# 📷 Captures de l’application

## Interface principale

![Interface](screenshots/page1.png)

## Recherche de nom de famille

![Recherche nom](screenshots/page2.png)

## Recherche de prénom

![Recherche prénom](screenshots/page3.png)

## Visualisations

![Visualisation 1](screenshots/page4.png)

![Visualisation 2](screenshots/page5.png)

---

# 🛠 Technologies utilisées

* **Python**
* **Streamlit**
* **SentenceTransformers**
* **Hugging Face Transformers**
* **scikit-learn**
* **ROUGE Score**
* **BeautifulSoup**
* **Requests**
* **NetworkX**
* **Matplotlib**
* **Plotly**
* **PyTorch**
* **JSON**

---

# 💡 Compétences mobilisées

Ce projet met en œuvre des compétences pratiques en :

* conception de pipeline NLP
* prétraitement et nettoyage de texte
* résumé extractif
* résumé avec transformers
* benchmark et évaluation de modèles
* recherche sémantique avec embeddings
* web scraping et parsing HTML
* extraction d’information avec regex et règles métier
* structuration de données en JSON
* visualisation de données
* développement d’application Streamlit

---

# 📌 Cas d’usage possibles

Ce projet peut être utile pour :

* l’exploration généalogique
* la recherche d’origine de noms et prénoms
* l’expérimentation autour du résumé automatique
* l’apprentissage de la recherche sémantique
* la création d’un produit data à partir de textes web

---

# ⚠️ Limites

* l’évaluation repose sur un nombre limité de résumés de référence
* le scraping des prénoms dépend de la structure du site source
* la qualité des résumés dépend de la richesse des textes d’origine
* certaines origines ou significations peuvent être incomplètes
* la summarization des prénoms est plus simple que celle des patronymes
* la similarité sémantique dépend du modèle d’embedding utilisé

---

# 🔮 Améliorations possibles

Améliorations possibles du projet :

* ajouter davantage de sources de données
* augmenter le nombre de résumés de référence pour l’évaluation
* enrichir la logique de résumé pour les prénoms
* ajouter un fuzzy matching en complément de la recherche sémantique
* améliorer la normalisation linguistique
* stocker les résultats dans une base de données
* déployer l’application en ligne
* ajouter des tests automatisés et un meilleur logging

---

# 👩‍💻 Auteur

**Eya BEN SALEM**
Projet réalisé dans le cadre du **Master Big Data & Intelligence Artificielle**.

---

# ⭐ Remarque finale

Ce projet illustre comment combiner :

* le **web scraping**
* la **structuration de données**
* le **résumé automatique**
* la **recherche sémantique**
* l’**évaluation NLP**
* la **visualisation**
* le **développement d’application interactive**

dans un pipeline complet, réutilisable et orienté produit.

