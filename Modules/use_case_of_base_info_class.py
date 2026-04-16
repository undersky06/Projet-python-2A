from Modules.base_info_class import ArticleBaseQuery
import pandas as pd
from collections import Counter
import re


class ArticleService:
    def __init__(self, dao: ArticleBaseQuery):
        self.dao = dao

    def get_articles_by_theme(self, theme_keywords: list[str]):
        pass

    def get_all_articles(self):
        """Get all articles as a dataframe"""
        return self.dao.get_all_articles()

    # Flexible filtering system for articles
    def filter_articles(self, *filters, **kwargs) -> pd.DataFrame:
        """
        Flexible filtering system:
        - kwargs = column == value filters
        - args = advanced lambda filters
        """
        return self.dao.filter_data(*filters, **kwargs)

    # counting functions
    def count_articles_by_year(self):
        """Count articles by publication year"""
        df = self.dao.get_all_articles(columns=["year"])
        return df["year"].value_counts().sort_index()

    def count_articles_by_study_type(self):
        """Count articles by study type"""
        df = self.dao.get_all_articles(columns=["study_type"])
        return df["study_type"].value_counts().sort_index()

    def count_articles_by_location(self):
        """Count articles by study location"""
        df = self.dao.get_all_articles(columns=["study_location"])
        return df["study_location"].value_counts().sort_index()

    def count_articles_by_data_source(self):
        """Count articles by data source"""
        df = self.dao.get_all_articles(columns=["data_source"])
        return df["data_source"].value_counts().sort_index()

    def count_word_frequencies(
        self,
        text_column: str = "abstract",
        top_n: int = 20,
        *args, **kwargs
    ):
        """Count word frequencies in a specific text column"""
        df = self.filter_articles(*args, **kwargs)
        # df = df[text_column]
        all_text = " ".join(df[text_column].dropna().tolist())
        words = all_text.split()
        word_freq = Counter(words)
        return dict(word_freq.most_common(top_n))

    def count_articles_by_keyword(self, keyword: list[str]):
        """Count articles containing a specific keyword in the keywords column"""
        df = self.dao.get_all_articles(columns=["keywords"])
        count = 0
        for kw_list in df["keywords"].dropna():
            if any(kw.strip() in kw_list.split(",") for kw in keyword):
                count += 1
        return count

    # Additional getters for specific fields with optional filtering
    def get_study_keywords(self, title: str = None, doi: str = None) -> set[str]:
        """Get unique study keywords from the articles"""
        return self.dao.get_study_keywords(title=title, doi=doi)

    def get_study_locations(self, title: str = None, doi: str = None) -> set[str]:
        """Get unique study locations from the articles"""
        return self.dao.get_study_locations(title=title, doi=doi)

    def get_study_years(self, title: str = None, doi: str = None) -> set[int]:
        """Get unique study years from the articles (both start and end years)"""
        start_years = self.dao.get_study_start_years(title=title, doi=doi)
        end_years = self.dao.get_study_end_years(title=title, doi=doi)
        return set(start_years).union(set(end_years))

    def get_study_data_sources(self, title: str = None, doi: str = None) -> set[str]:
        """Get unique data sources from the articles"""
        return self.dao.get_data_sources(title=title, doi=doi)

    def get_study_types(self, title: str = None, doi: str = None) -> set[str]:
        """Get unique study types from the articles"""
        return self.dao.get_study_types(title=title, doi=doi)

    def get_study_aims(self, title: str = None, doi: str = None) -> set[str]:
        """Get unique study aims from the articles"""
        return self.dao.get_study_aims(title=title, doi=doi)

    def get_study_abstracts(self, prefer_doi=True, *args, **kwargs) -> pd.DataFrame:
        """Get unique study abstracts from the articles"""
        self.filter_articles(*args, **kwargs)
        results = self.dao.get_study_absracts(prefer_doi=prefer_doi)
        return results

    # cleaning text data for NLP tasks
    def clean_text(self, text: str) -> str:
        """Basic text cleaning function"""
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"\s+", " ", text)  # Replace multiple whitespace with single space
        text = re.sub(r"\S@\S+", " ", text) # Remove email addresses
        text = re.sub(r"http\S+", " ", text) # Remove URLs
        text= re.sub(r"<.*?>", " ", text) # Remove HTML tags
        return text.strip()

    def apply_clean_text_data(self, text_column: str = "abstract", *args, **kwargs) -> pd.DataFrame:
        """Clean text data in a specific column for NLP tasks"""
        df = self.filter_articles(*args, **kwargs)
        print(df.columns)
        df[text_column] = df[text_column].apply(lambda text: self.clean_text(text))
        return df
