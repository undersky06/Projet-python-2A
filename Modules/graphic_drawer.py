from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


class GraphicDrawer:
    """Utility class for drawing common plots."""

    @staticmethod
    def _to_dataframe(data: Union[pd.DataFrame, dict]) -> pd.DataFrame:
        """Convert dict to DataFrame if necessary."""
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
        """Plot a bar plot."""
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
        """Plot a line plot."""
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
        """Plot a pie chart."""
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
        """Plot a histogram."""
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
        """Plot a box plot."""
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
        """Plot a scatter plot."""
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
        """Generate and display a word cloud."""
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
        wordcloud = WordCloud().generate_from_frequencies(freq)

        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        if title:
            plt.title(title)

        plt.tight_layout()
        plt.show()
