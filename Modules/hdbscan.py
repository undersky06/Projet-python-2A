"""
HDBSCAN + UMAP Clusterer
========================

This module provides a high-level wrapper combining UMAP for nonlinear
dimension reduction and HDBSCAN for density-based clustering. It includes
a robust composite scoring mechanism integrating cluster validity,
stability, and noise proportion.

Classes
-------
HdbscanClusterer
    Perform UMAP reduction followed by HDBSCAN clustering, with evaluation
    metrics and result export utilities.

Dependencies
------------
- hdbscan
- numpy
- pandas
- scikit-learn
- UmapDimensionReducer (local module)

Example
-------
>>> clusterer = HdbscanClusterer(metric="euclidean")
>>> clusterer.fit(
...     data=embeddings,
...     n_neighbors=15,
...     n_components=2,
...     min_cluster_size=30
... )
>>> clusterer.summary()
{'best_params': {...}, 'scores': {...}, 'n_clusters': 4}
>>> df = clusterer.get_results()
"""

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from Modules.umap import UmapDimensionReducer


class HdbscanClusterer:
    """
    HDBSCAN clustering with UMAP preprocessing.

    This class performs:
    1. UMAP dimension reduction
    2. HDBSCAN clustering
    3. Evaluation using DBCV or silhouette score
    4. Composite scoring combining validity, stability, and noise ratio

    Parameters
    ----------
    metric : str, optional (default="euclidean")
        Distance metric used by HDBSCAN.
    random_state : int, optional (default=42)
        Seed for reproducibility.

    Attributes
    ----------
    best_model : hdbscan.HDBSCAN or None
        Fitted HDBSCAN model.
    best_labels : ndarray or None
        Cluster labels assigned by HDBSCAN.
    best_params : dict or None
        Dictionary of parameters used for the best model.
    best_scores : dict or None
        Dictionary containing:
        - ``metric_val`` : DBCV or silhouette score
        - ``noise_ratio`` : proportion of points labeled as noise
        - ``stability`` : mean cluster persistence
        - ``composite_score`` : final aggregated score
    """

    def __init__(
        self,
        metric="euclidean",
        random_state=42,
    ):
        self.metric = metric
        self.random_state = random_state

        self.best_model = None
        self.best_labels = None
        self.best_params = None
        self.best_scores = None

    # -----------------------------------------------------
    # FIT DIRECT
    # -----------------------------------------------------
    def fit(
        self,
        data,
        n_neighbors: int,
        n_components: int,
        min_cluster_size: int,
        min_dist: float = 0.1,
        cluster_selection_method: str = "eom",
        metric_eval: str = "dbcv",
        min_samples_ratio: float = 0.5,
    ):
        """
        Fit UMAP + HDBSCAN without Bayesian Optimization.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input high-dimensional data.
        n_neighbors : int
            Number of neighbors for UMAP.
        n_components : int
            Target dimensionality for UMAP.
        min_cluster_size : int
            Minimum cluster size for HDBSCAN.
        min_dist : float, optional (default=0.1)
            UMAP minimum distance parameter.
        cluster_selection_method : {"eom", "leaf"}, optional
            HDBSCAN cluster selection strategy.
        metric_eval : {"dbcv", "silhouette"}, optional
            Metric used to evaluate clustering quality.
        min_samples_ratio : float, optional (default=0.5)
            Ratio used to compute ``min_samples = max(2, int(min_cluster_size * ratio))``.

        Returns
        -------
        self : HdbscanClusterer
            The fitted instance.

        Notes
        -----
        Composite score is computed as::

            composite = 0.5 * metric_val + 0.5 * stability - 0.35 * noise_ratio
        """

        # ------------------------
        # UMAP
        # ------------------------
        reducer = UmapDimensionReducer(
            data=data,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=self.random_state,
        )
        X_umap = reducer.fit_transform()

        # ------------------------
        # HDBSCAN
        # ------------------------
        min_samples = max(2, int(min_cluster_size * min_samples_ratio))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=self.metric,
            cluster_selection_method=cluster_selection_method,
            gen_min_span_tree=True,
        )

        labels = clusterer.fit_predict(X_umap)

        # ------------------------
        # Evaluation
        # ------------------------
        if metric_eval.lower() == "dbcv":
            metric_val = clusterer.relative_validity_
        else:
            mask = labels != -1
            if len(set(labels[mask])) <= 1:
                metric_val = -1
            else:
                metric_val = silhouette_score(X_umap[mask], labels[mask])

        noise_ratio = np.mean(labels == -1)

        stability = (
            np.mean(clusterer.cluster_persistence_)
            if len(clusterer.cluster_persistence_) > 0
            else 0.0
        )

        # ------------------------
        # Score composite robuste
        # ------------------------
        composite_score = 0.5 * metric_val + 0.5 * stability - 0.35 * noise_ratio

        # ------------------------
        # Stockage
        # ------------------------
        self.best_model = clusterer
        self.best_labels = labels
        self.best_params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "n_components": n_components,
            "cluster_selection_method": cluster_selection_method,
        }

        self.best_scores = {
            "metric_val": metric_val,
            "noise_ratio": noise_ratio,
            "stability": stability,
            "composite_score": composite_score,
        }

        return self

    # -----------------------------------------------------
    # Résumé
    # -----------------------------------------------------
    def summary(self):
        """
        Return a summary of the best clustering results.

        Returns
        -------
        dict
            Contains:
            - ``best_params`` : parameters used
            - ``scores`` : evaluation metrics
            - ``n_clusters`` : number of non-noise clusters

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self.best_labels is None:
            raise RuntimeError("Appeler fit() avant summary().")

        return {
            "best_params": self.best_params,
            "scores": self.best_scores,
            "n_clusters": len(set(self.best_labels))
            - (1 if -1 in self.best_labels else 0),
        }

    # -----------------------------------------------------
    # DataFrame résultats
    # -----------------------------------------------------
    def get_results(self, index=None):
        """
        Return a DataFrame containing cluster assignments.

        Parameters
        ----------
        index : array-like, optional
            Custom index for the output DataFrame. If None, uses ``range(n_samples)``.

        Returns
        -------
        pandas.DataFrame
            Columns:
            - ``id`` : index
            - ``cluster`` : assigned cluster label

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self.best_labels is None:
            raise RuntimeError("Appeler fit() avant get_results().")

        if index is None:
            index = range(len(self.best_labels))

        return pd.DataFrame({"id": index, "cluster": self.best_labels})
