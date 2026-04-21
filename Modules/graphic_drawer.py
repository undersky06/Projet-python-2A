"""
Graphic Drawing Utilities
=========================

This module provides a collection of static methods for generating common
visualizations using Matplotlib, Seaborn, and WordCloud. It supports:

- Bar plots
- Line plots
- Pie charts
- Histograms
- Box plots
- Scatter plots
- Word clouds

The module accepts either dictionaries or DataFrames for categorical plots,
and Pandas Series for numeric plots.

Classes
-------
GraphicDrawer
    Utility class containing static methods for drawing visualizations.

Dependencies
------------
- matplotlib
- seaborn
- pandas
- wordcloud

Example
-------
>>> data = {"A": 10, "B": 5, "C": 7}
>>> GraphicDrawer.draw_bar_plot(data, title="Example Bar Plot")

>>> text = "data science machine learning ai ai ai"
>>> GraphicDrawer.draw_wordcloud(text)
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


class GraphicDrawer:
    """Utility class for drawing common plots."""

    @staticmethod
    def _to_dataframe(data: Union[pd.DataFrame, dict]) -> pd.DataFrame:
        """
        Convert input data to a DataFrame.

        Parameters
        ----------
        data : dict or pandas.DataFrame
            Input data. If a dictionary is provided, it must map
            categories to counts.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ["Category", "Count"].

        Notes
        -----
        This helper ensures consistent formatting for plotting functions.
        """
        if isinstance(data, dict):
            return pd.DataFrame(list(data.items()), columns=["Category", "Count"])
        return data

    @staticmethod
    def draw_bar_plot(
        data: Union[pd.DataFrame, dict],
        top_n: int = 20,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple = (10, 6),
        grid: bool = True,
    ) -> None:
        """
        Draw a bar plot from categorical data.

        Parameters
        ----------
        data : dict or pandas.DataFrame
            Input data mapping categories to counts.
        top_n : int, optional (default=20)
            Number of top categories to display.
        title : str, optional
            Plot title.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        figsize : tuple, optional (default=(10, 6))
            Figure size.
        grid : bool, optional (default=True)
            Whether to display a grid.

        Returns
        -------
        None
        """
        df = GraphicDrawer._to_dataframe(data).head(top_n)

        plt.figure(figsize=figsize)
        sns.barplot(data=df, x="Count", y="Category")

        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_line_plot(
        data: Union[pd.DataFrame, dict],
        top_n: int = 20,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple = (10, 6),
        grid: bool = True,
    ) -> None:
        """
        Draw a line plot from categorical data.

        Parameters
        ----------
        data : dict or pandas.DataFrame
            Input data mapping categories to counts.
        top_n : int, optional
            Number of top categories to display.
        title : str, optional
            Plot title.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        figsize : tuple, optional
            Figure size.
        grid : bool, optional
            Whether to display a grid.

        Returns
        -------
        None
        """
        df = GraphicDrawer._to_dataframe(data).head(top_n)

        plt.figure(figsize=figsize)
        sns.lineplot(data=df, x="Category", y="Count", marker="o")

        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_pie_chart(
        data: Union[pd.DataFrame, dict],
        top_n: int = 20,
        title: Optional[str] = None,
        figsize: tuple = (8, 8),
    ) -> None:
        """
        Draw a pie chart from categorical data.

        Parameters
        ----------
        data : dict or pandas.DataFrame
            Input data mapping categories to counts.
        top_n : int, optional
            Number of top categories to display.
        title : str, optional
            Plot title.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        None
        """
        df = GraphicDrawer._to_dataframe(data).head(top_n)

        plt.figure(figsize=figsize)
        plt.pie(
            df["Count"],
            labels=df["Category"],
            autopct="%1.1f%%",
            startangle=140,
        )

        if title:
            plt.title(title)

        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_histogram(
        data: pd.Series,
        bins: int = 10,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple = (10, 6),
        grid: bool = True,
    ) -> None:
        """
        Draw a histogram for numeric data.

        Parameters
        ----------
        data : pandas.Series
            Numeric data to plot.
        bins : int, optional (default=10)
            Number of histogram bins.
        title : str, optional
            Plot title.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        figsize : tuple, optional
            Figure size.
        grid : bool, optional
            Whether to display a grid.

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        plt.hist(data, bins=bins, edgecolor="black")

        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_box_plot(
        data: pd.Series,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple = (10, 6),
        grid: bool = True,
    ) -> None:
        """
        Draw a box plot for numeric data.

        Parameters
        ----------
        data : pandas.Series
            Numeric data to plot.
        title : str, optional
            Plot title.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        figsize : tuple, optional
            Figure size.
        grid : bool, optional
            Whether to display a grid.

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        sns.boxplot(x=data)

        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_scatter_plot(
        x: pd.Series,
        y: pd.Series,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple = (10, 6),
        grid: bool = True,
    ) -> None:
        """
        Draw a scatter plot.

        Parameters
        ----------
        x : pandas.Series
            X-axis values.
        y : pandas.Series
            Y-axis values.
        title : str, optional
            Plot title.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        figsize : tuple, optional
            Figure size.
        grid : bool, optional
            Whether to display a grid.

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        sns.scatterplot(x=x, y=y)

        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_wordcloud(
        text: str,
        max_words: int = 200,
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
        background_color: str = "white",
    ) -> None:
        """
        Generate and display a word cloud.

        Parameters
        ----------
        text : str
            Input text from which to generate the word cloud.
        max_words : int, optional (default=200)
            Maximum number of words to include.
        title : str, optional
            Plot title.
        figsize : tuple, optional
            Figure size.
        background_color : str, optional (default="white")
            Background color of the word cloud.

        Returns
        -------
        None
        """
        wordcloud = WordCloud(
            max_words=max_words,
            background_color=background_color,
        ).generate(text)

        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        if title:
            plt.title(title)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_wordcloud_from_freq(freq, title=None, figsize=(10, 6)):
        """
        Generate and display a word cloud from word frequencies.

        Parameters
        ----------
        freq : dict
            Dictionary mapping words to their frequencies.
            Example: {"data": 12, "science": 8, "ai": 5}
        title : str, optional
            Title of the plot.
        figsize : tuple, optional (default=(10, 6))
            Size of the figure.

        Returns
        -------
        None

        Notes
        -----
        This function expects a frequency dictionary already computed,
        for example using TextProcessor.count_word_frequencies().
        """
        wordcloud = WordCloud().generate_from_frequencies(freq)

        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        if title:
            plt.title(title)

        plt.tight_layout()
        plt.show()
