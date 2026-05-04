"""
analysis.py

Functions for analyzing crime datasets.
"""

import pandas as pd


def total_crime_by_year(df, year_column, value_column):
    """
    Compute total crime per year.
    """
    return (
        df.groupby(year_column)[value_column]
        .sum()
        .reset_index()
    )


def total_crime_by_county(df, county_column, value_column):
    """
    Compute total crime per county.
    """
    return (
        df.groupby(county_column)[value_column]
        .sum()
        .reset_index()
    )


def average_crime(df, value_column):
    """
    Compute average crime value.
    """
    return df[value_column].mean()
