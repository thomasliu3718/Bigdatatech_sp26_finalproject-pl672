"""
cleaning.py

Functions for cleaning New York crime datasets.
"""

import pandas as pd


def standardize_column_names(df):
    """
    Standardize column names by stripping spaces, lowering case,
    and replacing spaces with underscores.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = (
        cleaned_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return cleaned_df


def remove_missing_rows(df):
    """
    Remove rows with any missing values.
    """
    return df.dropna().reset_index(drop=True)


def filter_by_year_range(df, year_column, start_year, end_year):
    """
    Filter dataset by a year range.
    """
    cleaned_df = df.copy()
    return cleaned_df[
        (cleaned_df[year_column] >= start_year) &
        (cleaned_df[year_column] <= end_year)
    ].reset_index(drop=True)
