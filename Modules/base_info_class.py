import functools
from typing import List, Optional

import pandas as pd

REQUIRED_FIELDS = {
    "title",
    "abstract",
    "authors",
    "study_aim",
    "study_location",
    "study_year_start",
    "study_year_end",
    "study_type",
    "data_source",
}


class ArticleBaseQuery:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._df_cache: Optional[pd.DataFrame] = None

    def _load_dataframe(self) -> pd.DataFrame:
        """Load dataframe with caching"""
        if self._df_cache is not None:
            return self._df_cache

        if self.file_path.endswith(".csv"):
            self._df_cache = pd.read_csv(self.file_path)
        elif self.file_path.endswith((".xlsx", ".xls")):
            self._df_cache = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format")

        return self._df_cache

    def add_authors(self, authors):
        """Add authors column if missing"""
        self._df_cache["authors"] = authors

    def _normalize_value(self, value):
        if pd.isna(value):
            return None
        return value

    # wrappers for ensuring immutability and type safety
    def ensure_dataframe(func):
        """Decorator to ensure the method returns a DataFrame"""

        @functools.wraps(func)  # Preserve function metadata (name, docstring)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if not isinstance(result, pd.DataFrame):
                raise TypeError("Expected a DataFrame")
            return result

        return wrapper

    def ensure_columns_exist(columns: List[str]):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                df = self._load_dataframe()
                df_columns = set(df.columns)
                missing = set(columns) - df_columns

                if missing:
                    raise ValueError(
                        f"Missing required columns in data: {', '.join(sorted(missing))}"
                    )

                return func(self, *args, **kwargs)

            return wrapper

        return decorator

    def validate_required_fields(func):
        """Decorator to ensure reqquired fields are present in the dataframe"""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            df = self._load_dataframe()
            self._validate_required_fields(df)
            return func(self, *args, **kwargs)

        return wrapper

    def _validate_required_fields(self, df: pd.DataFrame):
        """Validate that required fields exist in dataframe"""
        missing_columns = REQUIRED_FIELDS - set(df.columns)
        if missing_columns:
            raise ValueError(
                "Missing required columns in data:"
                f" {', '.join(sorted(missing_columns))}"
            )

    def display_columns(self):
        """Utility method to display columns in the dataframe"""
        df = self._load_dataframe()
        print(f"Columns in data: {list(df.columns)}")

    # getters with optional filtering by title or doi
    # @validate_required_fields
    @ensure_dataframe
    def get_all_articles(self, columns: List[str] = None) -> pd.DataFrame:
        """Get all articles as a DataFrame"""
        if columns is not None:
            return self.select_columns(columns)
        return self._load_dataframe()

    @validate_required_fields
    def get_study_types(self, title: str = None, doi: str = None) -> set[str]:
        """Get unique study types from the articles"""
        df = self._load_dataframe()
        if title is not None and doi is None:
            df = df[df["title"].str.contains(title, case=False, na=False)]
        if doi is not None:
            df = df[df["doi"] == doi]
        study_types = df["study_type"].dropna().unique()
        return study_types

    @validate_required_fields
    @ensure_dataframe
    def get_study_aims(self, title: str = None) -> set[str]:
        """Get unique study aims from the articles"""
        df = self._load_dataframe()
        if title is not None:
            df = df[df["title"].str.contains(title, case=False, na=False)]
        study_aims = df["study_aim"].dropna().unique()
        return study_aims

    @validate_required_fields
    @ensure_dataframe
    def get_study_locations(self, title: str = None, doi: str = None) -> set[str]:
        """Get unique study locations from the articles"""
        df = self._load_dataframe()
        if title is not None and doi is None:
            df = df[df["title"].str.contains(title, case=False, na=False)]
        if doi is not None:
            df = df[df["doi"] == doi]
        study_locations = df["study_location"].dropna().unique()
        return study_locations

    @validate_required_fields
    @ensure_dataframe
    def get_study_start_years(self, title: str = None, doi: str = None) -> set[int]:
        """Get unique study start years from the articles"""
        df = self._load_dataframe()
        if title is not None and doi is None:
            df = df[df["title"].str.contains(title, case=False, na=False)]
        if doi is not None:
            df = df[df["doi"] == doi]
        study_years = df["study_year_start"].dropna().unique()
        return study_years

    @validate_required_fields
    @ensure_dataframe
    def get_study_end_years(self, title: str = None, doi: str = None) -> set[int]:
        """Get unique study end years from the articles"""
        df = self._load_dataframe()
        if title is not None and doi is None:
            df = df[df["title"].str.contains(title, case=False, na=False)]
        if doi is not None:
            df = df[df["doi"] == doi]
        study_years = df["study_year_end"].dropna().unique()
        return study_years

    @validate_required_fields
    @ensure_dataframe
    def get_data_sources(self, title: str = None, doi: str = None) -> set[str]:
        """Get unique data sources from the articles"""
        df = self._load_dataframe()
        if title is not None and doi is None:
            df = df[df["title"].str.contains(title, case=False, na=False)]
        if doi is not None:
            df = df[df["doi"] == doi]
        data_sources = df["data_source"].dropna().unique()
        return data_sources

    @validate_required_fields
    @ensure_dataframe
    def get_study_keywords(self, title: str = None, doi: str = None) -> set[str]:
        """Get unique study keywords from the articles"""
        df = self._load_dataframe()
        if title is not None and doi is None:
            df = df[df["title"].str.contains(title, case=False, na=False)]
        if doi is not None:
            df = df[df["doi"] == doi]
        keywords_series = df["keywords"].dropna().unique()
        keywords_set = set()
        for kw in keywords_series:
            for k in kw.split(","):
                keywords_set.add(k.strip())
        return keywords_set

    @validate_required_fields
    @ensure_dataframe
    def get_study_absracts(self, prefer_doi: bool = True) -> set[str]:
        """Get unique study abstracts from the articles"""
        key = "doi" if prefer_doi else "title"

        abstracts_df = self.select_columns([key, "abstract"])
        abstracts_df = abstracts_df.dropna(subset=[key, "abstract"])
        abstracts_df = abstracts_df.drop_duplicates(subset=[key, "abstract"])
        return abstracts_df

    # selecting columns
    @validate_required_fields
    @ensure_dataframe
    def select_columns(self, columns: List[str]) -> pd.DataFrame:
        """Select specific columns from the dataframe"""
        df = self._load_dataframe()
        missing_columns = set(columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                "Missing columns in data:" f" {', '.join(sorted(missing_columns))}"
            )
        return df[columns]

    # advanced filtering and analysis
    @validate_required_fields
    @ensure_dataframe
    def filter_data(self, *filters, **kfilters) -> pd.DataFrame:
        """Filter dataframe using flexible conditions"""

        df = self._load_dataframe()

        # check that all kfilters keys are valid columns
        missing = set(kfilters.keys()) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for col, value in kfilters.items():
            df = df[df[col] == value]
        # advanced filters are functions that take
        # the dataframe and return a boolean mask
        # may be use with lambda functions for complex conditions
        for f in filters:
            df = df[f(df)]

        return df

    @validate_required_fields
    @ensure_dataframe
    def build_nlp_text(self, inplace: bool = False) -> pd.DataFrame:
        """Build a text field for NLP vectorization by concatenating relevant fields"""
        if not inplace:
            df = self._load_dataframe()
            df["nlp_text"] = (
                df["title"].fillna("")
                + " "
                + df["abstract"].fillna("")
                + " "
                + df["study_aim"].fillna("")
                + " "
                + df["keywords"].fillna("")
            )
            return df
        else:
            df = self._load_dataframe()
            df["nlp_text"] = (
                df["title"].fillna("")
                + " "
                + df["abstract"].fillna("")
                + " "
                + df["study_aim"].fillna("")
                + " "
                + df["keywords"].fillna("")
            )
            self._df_cache = df
