"""
UMAP Dimension Reduction Utility
===============================

This module provides a high-level wrapper around UMAP for dimensionality
reduction of labeled or unlabeled embeddings. It supports:

- DataFrames with custom indices (e.g., image IDs)
- Normalization before reduction
- 2D or 3D interactive visualization using Plotly

Classes
-------
UmapDimensionReducer
    Perform UMAP reduction and optional visualization.

Dependencies
------------
- pandas
- plotly
- umap-learn
- scikit-learn (normalize)

Example
-------
>>> reducer = UmapDimensionReducer(
...     n_neighbors=15,
...     n_components=2,
...     data=df_embeddings
... )
>>> X_2d = reducer.fit_transform()
>>> fig = reducer.plot(dims=2, labels="index")
>>> fig.show()
"""

import pandas as pd
import plotly.express as px
import umap
from sklearn.preprocessing import normalize


class UmapDimensionReducer:
    """
    UMAP-based dimensionality reduction for embedding matrices.

    Parameters
    ----------
    n_neighbors : int, optional (default=10)
        Number of neighbors used by UMAP to build the local manifold structure.
        Must be a strictly positive integer.
    min_dist : float, optional (default=0.1)
        Minimum distance between embedded points. Must be in [0, 1].
    metric : str, optional (default="cosine")
        Distance metric used by UMAP.
    n_components : int, optional (default=10)
        Target dimensionality of the reduced space.
    random_state : int, optional (default=42)
        Seed for reproducibility.
    data : pandas.DataFrame or None, optional
        Input embedding matrix. If provided as a DataFrame, its index is preserved
        and used in visualizations.

    Attributes
    ----------
    reducer : umap.UMAP or None
        Internal UMAP model instance.
    _X_reduced : numpy.ndarray or None
        Reduced embedding matrix after calling `fit_transform`.
    data : pandas.DataFrame or None
        Original input data.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        min_dist: float = 0.1,
        metric: str = "cosine",
        n_components: int = 10,
        random_state: int = 42,
        data: pd.DataFrame | None = None,
    ):
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("`n_neighbors` doit être un entier strictement positif.")

        if not isinstance(min_dist, (int, float)) or not (0.0 <= min_dist <= 1.0):
            raise ValueError("`min_dist` doit être un réel compris entre 0 et 1.")

        self.n_neighbors = n_neighbors
        self.min_dist = float(min_dist)
        self.metric = metric
        self.n_components = n_components
        self.random_state = random_state
        self.data = data
        self.reducer = None
        self._X_reduced = None

    def reduce(self):
        """
        Initialize the UMAP reducer with the configured parameters.

        Returns
        -------
        self : UmapDimensionReducer
            The instance itself, with the UMAP reducer created.

        Notes
        -----
        This method does not perform any computation. It only instantiates
        the UMAP model. The actual reduction happens in `fit_transform`.
        """
        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric=self.metric,
            random_state=self.random_state,
            min_dist=self.min_dist,
        )
        return self

    def fit_transform(self):
        """
        Apply UMAP dimensionality reduction to the input data.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_components)
            The reduced embedding matrix.

        Raises
        ------
        ValueError
            If no data has been provided.
        RuntimeError
            If UMAP reducer has not been initialized (rare, auto-handled).

        Notes
        -----
        - Input data is L2-normalized before reduction.
        - If `reduce()` has not been called, it is invoked automatically.
        """
        if self.reducer is None:
            self.reduce()

        if self.data is None:
            raise ValueError("Aucune donnée fournie pour la réduction.")

        X = self.data.values  # on récupère emb_matrix
        X = normalize(X, norm="l2")

        self._X_reduced = self.reducer.fit_transform(X)
        return self._X_reduced

    def plot(
        self,
        dims: int = 2,
        labels: str | None = None,
        title: str = "Projection UMAP",
    ):
        """
        Visualize the UMAP projection using Plotly.

        Parameters
        ----------
        dims : int, optional (default=2)
            Number of dimensions to plot (2 or 3).
        labels : str or None, optional
            Label source for coloring points:
            - "index" : use DataFrame index
            - column name : use a column from `data`
            - None : no coloring
        title : str, optional (default="Projection UMAP")
            Title of the Plotly figure.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive scatter plot (2D or 3D).

        Raises
        ------
        RuntimeError
            If `fit_transform` has not been called.
        ValueError
            If `dims` is not 2 or 3.
            If `labels` is invalid.

        Notes
        -----
        - The DataFrame index is used as hover labels.
        - For 3D visualization, the first three UMAP components are used.
        """
        if self._X_reduced is None:
            raise RuntimeError("Appeler `fit_transform` avant l'affichage.")

        if dims not in (2, 3):
            raise ValueError("`dims` doit être 2 ou 3.")

        df = pd.DataFrame(
            self._X_reduced,
            index=self.data.index,
            columns=[f"UMAP-{i+1}" for i in range(self.n_components)],
        )

        color_col = None

        if labels is not None:
            if labels == "index":
                df["labels"] = df.index.astype(str)
                color_col = "labels"

            elif labels in self.data.columns:
                df["labels"] = self.data[labels].astype(str)
                color_col = "labels"

            else:
                raise ValueError(
                    f"Label '{labels}' introuvable dans les colonnes ou index."
                )

        if dims == 2:
            fig = px.scatter(
                df,
                x="UMAP-1",
                y="UMAP-2",
                color=color_col,
                hover_name=df.index.astype(str),
                opacity=0.7,
                title=title,
            )
        else:
            fig = px.scatter_3d(
                df,
                x="UMAP-1",
                y="UMAP-2",
                z="UMAP-3",
                color=color_col,
                hover_name=df.index.astype(str),
                opacity=0.7,
                title=title,
            )

        return fig
