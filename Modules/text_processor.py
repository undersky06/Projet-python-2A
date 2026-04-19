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
        cleaned = [self.preprocess(t) for t in texts]
        return self.vectorizer.fit_transform(cleaned)

    @ensure_valid_texts_list
    def transform_tfidf(self, texts: List[str]):
        cleaned = [self.preprocess(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    def get_features(self):
        return self.vectorizer.get_feature_names_out()

    # ----------------------------
    # SCI-BERT EMBEDDINGS
    # ----------------------------
    def _mean_pooling(self, outputs, attention_mask):
        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)

        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)

        return summed / counts

    @ensure_valid_texts_list  # noqa: F821
    def embed_scibert(self, texts: List[str], batch_size: int = 32):
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
