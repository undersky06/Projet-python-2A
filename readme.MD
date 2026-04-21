# Projet Python 2A

## Problématique

Ce projet porte sur l’application de techniques de **Traitement Automatique du Langage Naturel (NLP)** à un jeu de données d’articles scientifiques médicaux contenu dans le fichier `article_dat.csv`.

L’objectif principal est d’analyser et de structurer la littérature biomédicale à partir du contenu textuel (titres, résumés) et des variables associées aux études. Le dataset contient notamment des informations sur les types d’études, les journaux de publication et certaines caractéristiques méthodologiques.

Une attention particulière est portée aux thématiques impliquant des populations humaines, pouvant inclure des variables sensibles telles que l’ethnie, la race ou des caractéristiques démographiques. L’analyse est strictement méthodologique et statistique, sans interprétation clinique ou normative.

Le fichier `model_dat.csv` n’est pas utilisé dans ce projet.

---

## Objectifs

### 1. Statistiques descriptives
- Nombre d’articles par année  
- Distribution par journal  
- Distribution par type d’étude  
- Articles liés à la médecine de la reproduction  
- Exploration générale du dataset  

---

### 2. Exploration textuelle
- Nettoyage des textes (minuscules, ponctuation, stopwords)  
- Statistiques lexicales (fréquences, n-grammes)  
- Visualisations (nuages de mots, tendances temporelles)  

---

### 3. Représentation TF-IDF
- Construction de la matrice TF-IDF (titres et abstracts)  
- Identification des termes importants  
- Clustering thématique des articles  

---

### 4. Représentations avancées (optionnel)
- Embeddings BERT / Sentence-BERT  
- Calcul de similarité sémantique entre abstracts  
- Clustering dans l’espace des embeddings  

---

## Données

- Seul le fichier `article_dat.csv` est utilisé  
- Le fichier `model_dat.csv` est ignoré  

---

## Installation des dépendances

Installer les dépendances avec :

```bash
pip install -r requirements.txt
```

ou (recommandé) :

```bash
uv pip install -r requirements.txt
```

---

## Exécution du projet

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd Projet-python-2A
```

---

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

---

### 3. Activer l’environnement

Linux / Mac :
```bash
source .venv/bin/activate
```

Windows :
```bash
.venv\Scripts\activate
```

---

### 4. Installer les dépendances

```bash
uv pip install -r requirements.txt
```

---

### 5. Exécuter le projet

Script Python :

```bash
python main.py
```

Jupyter Notebook :

```bash
jupyter notebook
```

ou

```bash
jupyter lab
```

---

## 6. Configuration VSCode et Jupyter (IMPORTANT)

It is essential that all group members use the **correct Python interpreter**, otherwise execution errors or dependency issues may occur.

### VSCode setup

In VSCode:
- Open the command palette (`Ctrl + Shift + P`)
- Select: **Python: Select Interpreter**
- Choose:

```
.venv/bin/python
```

or the full path:

```
/home/<user>/work/Projet-python-2A/Projet-python-2A/.venv/bin/python
```

---

### Jupyter configuration

In a Jupyter notebook:

- Click **Kernel → Change Kernel**
- Select the kernel associated with:

```
Python (.venv)
```

If the kernel is not visible, run:

```bash
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

---

### Verification

To verify the active environment:

```bash
python -c "import sys; print(sys.executable)"
```

The output must point to `.venv`.

---

## Scope clarification

This project is strictly academic and methodological. It aims to analyze biomedical scientific literature using NLP techniques, including studies involving sensitive human attributes (such as ethnicity, race, and demographic characteristics), strictly for statistical and exploratory purposes.

---

## Contact

Group 2C

---

# English Version

## Problem Statement

This project applies **Natural Language Processing (NLP)** techniques to a dataset of biomedical scientific articles contained in `article_dat.csv`.

The main objective is to analyze and structure biomedical literature using textual content (titles, abstracts) and metadata related to study characteristics. The dataset includes information such as study type, publication journals, and methodological variables.

A particular focus is placed on studies involving human populations, which may include sensitive attributes such as ethnicity, race, or demographic characteristics. The analysis is strictly methodological and statistical, with no clinical or normative interpretation.

The file `model_dat.csv` is not used in this project.

---

## Objectives

### 1. Descriptive statistics
- Number of articles per year  
- Distribution by journal  
- Distribution by study type  
- Articles related to reproductive medicine  
- General dataset exploration  

---

### 2. Text exploration
- Text preprocessing (lowercasing, punctuation removal, stopwords removal)  
- Lexical statistics (word frequencies, n-grams)  
- Visualizations (word clouds, temporal trends)  

---

### 3. TF-IDF representation
- Construction of TF-IDF matrix (titles and abstracts)  
- Identification of important terms  
- Thematic clustering of articles  

---

### 4. Advanced representations (optional)
- BERT / Sentence-BERT embeddings  
- Semantic similarity between abstracts  
- Clustering in embedding space  

---

## Data

- Only `article_dat.csv` is used  
- `model_dat.csv` is ignored  

---

## Dependencies installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

or (recommended):

```bash
uv pip install -r requirements.txt
```

---

## Project execution

### 1. Clone repository

```bash
git clone <repo-url>
cd Projet-python-2A
```

---

### 2. Create virtual environment

```bash
python -m venv .venv
```

---

### 3. Activate environment

Linux / Mac:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

---

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

---

### 5. Run project

Python script:

```bash
python main.py
```

Jupyter Notebook:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

---

## 6. VSCode and Jupyter configuration (IMPORTANT)

It is essential that all group members use the correct Python interpreter to avoid execution and dependency issues.

### VSCode setup
- Open command palette (`Ctrl + Shift + P`)
- Select **Python: Select Interpreter**
- Choose `.venv/bin/python`

### Jupyter setup
- Select **Kernel → Change Kernel**
- Choose **Python (.venv)**

If missing:

```bash
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

---

## Verification

```bash
python -c "import sys; print(sys.executable)"
```

Must point to `.venv`.

---

## Scope clarification

This project is strictly academic and methodological. It analyzes biomedical scientific literature using NLP techniques, including studies involving sensitive attributes (ethnicity, race, demographic factors), strictly for exploratory and statistical analysis purposes.

---

## Contact

Group 2C