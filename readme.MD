# Project Python 2A

## Problem statement

This project focuses on Natural Language Processing (NLP) techniques applied to a dataset of scientific medical articles contained in `article_dat.csv`.

The main objective is to analyze and structure biomedical research literature using both textual content and associated metadata. The dataset contains article titles, abstracts, publication metadata, and variables related to study characteristics.

A particular focus is placed on medical research topics involving human populations, including studies that may reference sensitive attributes such as ethnicity, race, and demographic groups. The goal is strictly analytical and methodological: we aim to understand how these variables are represented in scientific literature and how they relate to research trends, without any normative or clinical interpretation.

We will not use `model_dat.csv` and will only rely on `article_dat.csv`.

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
- Text cleaning (lowercase, punctuation removal, stopwords)
- Lexical statistics (word frequencies, n-grams)
- Visualizations (word clouds, trends over time)

---

### 3. TF-IDF representation
- TF-IDF matrix on titles and abstracts
- Identification of important terms
- Thematic clustering of articles

---

### 4. Advanced embeddings (optional)
- BERT / Sentence-BERT embeddings
- Semantic similarity between abstracts
- Clustering of articles in embedding space

---

## Data

- Only `article_dat.csv` is used
- `model_dat.csv` is ignored

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

or (recommended):

```bash
uv pip install -r requirements.txt
```

Make sure you are inside the project directory and your virtual environment is activated.

---

## How to run the project

### 1. Clone the repository

```bash
git clone <url-du-repo>
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

### 5. Run the main script / notebooks

If using a Python script:

```bash
python main.py
```

If using Jupyter:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

---

### 6. VSCode configuration (important)

Select the interpreter:

```
.venv/bin/python
```

or full path:

```
/home/<user>/work/Projet-python-2A/Projet-python-2A/.venv/bin/python
```

---

### 7. Verify installation

```bash
python -c "import pandas; print('OK')"
```

---

## Scope clarification

This project is strictly academic and methodological. It aims to analyze biomedical scientific literature using NLP tools, including studies involving sensitive human attributes (such as ethnicity or race), strictly for statistical and textual analysis purposes.

---

## Contact

Group 2C