import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from Modules.umap import UmapDimensionReducer


class HdbscanClusterer:
    """
    Clusterisation avec HDBSCAN + UMAP
    Version avec fit direct et score composite robuste.
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
        Fit UMAP + HDBSCAN sans Bayesian Optimization.
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
        if self.best_labels is None:
            raise RuntimeError("Appeler fit() avant get_results().")

        if index is None:
            index = range(len(self.best_labels))

        return pd.DataFrame({"id": index, "cluster": self.best_labels})
