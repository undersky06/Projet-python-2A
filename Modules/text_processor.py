"""
Text Processing Utilities
=========================

This module provides text preprocessing, TF‑IDF vectorization, SciBERT
embeddings, and word‑frequency analysis. It includes robust input validation,
cleaning pipelines, and optional SciBERT integration for scientific text.

Decorators
----------
ensure_valid_texts_list
    Validate that the input is a list of non-empty strings.

Classes
-------
TextProcessor
    Preprocess text, compute TF‑IDF vectors, extract word frequencies,
    and generate SciBERT embeddings.

Dependencies
------------
- nltk
- numpy
- torch
- scikit-learn
- transformers
- rich (for progress bars)

Example
-------
>>> tp = TextProcessor(use_scibert=False)
>>> tokens = tp.preprocess("This is an example sentence.")
>>> tokens
['example', 'sentence']

>>> freqs = tp.count_word_frequencies("example example test")
>>> freqs
{'example': 2, 'test': 1}

>>> X = tp.fit_transform_tfidf(["text one", "text two"])
>>> X.shape
(2, n_features)
"""

import functools
import re
from collections import Counter
from typing import Dict, List, Union

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer


def ensure_valid_texts_list(func):
    """Decorator to validate the input 'texts' argument"""

    @functools.wraps(func)
    def wrapper(self, texts, *args, **kwargs):

        # ---- type check ----
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        # ---- content check ----
        cleaned_texts = []
        for t in texts:

            if t is None:
                raise ValueError("texts contains None")

            if not isinstance(t, str):
                raise TypeError("all elements in texts must be strings")

            if t.strip() == "":
                raise ValueError("texts contains empty strings")

            if isinstance(t, float) and np.isnan(t):
                raise ValueError("texts contains NaN")

            cleaned_texts.append(t)

        # call function with validated texts
        return func(self, cleaned_texts, *args, **kwargs)

    return wrapper


class TextProcessor:
    """
    Text processing pipeline supporting TF‑IDF and SciBERT embeddings.

    Parameters
    ----------
    use_scibert : bool, optional (default=False)
        If True, SciBERT tokenizer and model are loaded for embedding generation.
    download : bool, optional (default=False)
        If True, downloads required NLTK resources (stopwords, punkt).

    Attributes
    ----------
    stop_words : set
        English stopwords loaded from NLTK.
    vectorizer : TfidfVectorizer
        TF‑IDF vectorizer instance.
    tokenizer : transformers.AutoTokenizer, optional
        SciBERT tokenizer (only if use_scibert=True).
    model : transformers.AutoModel, optional
        SciBERT model (only if use_scibert=True).
    """

    def __init__(self, use_scibert: bool = False, download: bool = False):
        if download:
            nltk.download("stopwords")
            nltk.download("punkt")
            nltk.download("punkt_tab")

        self.stop_words = set(stopwords.words("english"))
        self.use_scibert = use_scibert

        self.vectorizer = TfidfVectorizer()

        if self.use_scibert:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "allenai/scibert_scivocab_uncased"
            )
            self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
            self.model.eval()

    # ----------------------------
    # CLEANING
    # ----------------------------
    def preprocess(self, text: str) -> Union[str, List[str]]:
        if not isinstance(text, str):
            return "" if self.use_scibert else []

        text = text.strip()

        # common cleaning (safe for both)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"\s+", " ", text)

        if self.use_scibert:
            # light cleaning only
            return text.strip()

        # TF-IDF mode (aggressive cleaning)
        text = text.lower()
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = word_tokenize(text)

        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 1]

        return tokens

    # ----------------------------
    # WORD FREQUENCIES
    # ----------------------------
    def count_word_frequencies(
        self,
        text: Union[str, List[str]],
        top_n: int = 20,
        preprocess: bool = True,
    ) -> Dict[str, int]:
        """
        Count the most frequent words in a text.

        Parameters
        ----------
        text : str or list of str
            Input text or token list.
        top_n : int, optional (default=20)
            Number of most frequent words to return.
        preprocess : bool, optional (default=True)
            Whether to apply preprocessing before counting.

        Returns
        -------
        dict
            Mapping word → frequency, sorted by frequency.

        Notes
        -----
        If text is a string, it is split on whitespace after preprocessing.
        """
        if preprocess:
            text = self.preprocess(text)

        if isinstance(text, list):
            words = text
        else:
            words = text.split()
        return dict(Counter(words).most_common(top_n))

    # ----------------------------
    # TF-IDF
    # ----------------------------
    @ensure_valid_texts_list
    def fit_transform_tfidf(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer and transform the input texts.

        Parameters
        ----------
        texts : list of str
            List of raw text documents.

        Returns
        -------
        scipy.sparse.csr_matrix
            TF-IDF matrix of shape (n_samples, n_features).
        """
        cleaned = [self.preprocess(t) for t in texts]
        return self.vectorizer.fit_transform(cleaned)

    @ensure_valid_texts_list
    def transform_tfidf(self, texts: List[str]):
        """
        Transform new texts using an already fitted TF-IDF vectorizer.

        Parameters
        ----------
        texts : list of str
            List of raw text documents.

        Returns
        -------
        scipy.sparse.csr_matrix
            TF-IDF matrix.
        """
        cleaned = [self.preprocess(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    def get_features(self):
        """
        Return the TF‑IDF feature names.

        Returns
        -------
        ndarray of str
            Array of feature names.
        """
        return self.vectorizer.get_feature_names_out()

    # ----------------------------
    # SCI-BERT EMBEDDINGS
    # ----------------------------
    def _mean_pooling(self, outputs, attention_mask):
        """
        Apply mean pooling over SciBERT token embeddings.

        Parameters
        ----------
        outputs : transformers.modeling_outputs.BaseModelOutput
            Output of the SciBERT model.
        attention_mask : torch.Tensor
            Attention mask of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Sentence embeddings of shape (batch_size, hidden_dim).

        Notes
        -----
        Mean pooling is computed as:

            sum(token_embeddings * mask) / sum(mask)

        Mask ensures padding tokens do not contribute.
        """
        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)

        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)

        return summed / counts

    @ensure_valid_texts_list  # noqa: F821
    def embed_scibert(self, texts: List[str], batch_size: int = 32):
        """
        Generate SciBERT embeddings for a list of texts.

        Parameters
        ----------
        texts : list of str
            List of raw text documents. Each text undergoes light cleaning
            before tokenization to preserve scientific terminology.

        Returns
        -------
        np.ndarray of shape (n_samples, hidden_dim)
            Dense matrix of sentence embeddings obtained via mean pooling
            over SciBERT token embeddings.

        Raises
        ------
        RuntimeError
            If SciBERT is not enabled (use_scibert=False).
        ValueError
            If the input list is empty.

        Notes
        -----
        - Uses the SciBERT model ``allenai/scibert_scivocab_uncased``.
        - Embeddings are computed as:

            embedding = sum(token_embeddings * mask) / sum(mask)

        - Padding tokens are excluded from the mean via the attention mask.
        - Returned embeddings are moved to CPU and converted to NumPy arrays.
        """
        if not self.use_scibert:
            raise ValueError("SciBERT is not enabled. Set use_scibert=True.")

        embeddings = []

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "[green] Computing SciBert embeddings ...", total=len(texts)
            )

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)

                batch_embeddings = self._mean_pooling(
                    outputs,
                    inputs["attention_mask"],
                )

                embeddings.append(batch_embeddings.cpu().numpy())

                progress.update(task, advance=len(batch))

        return np.vstack(embeddings)
