"""
visualization.py

Functions for visualizing crime data.
"""

import matplotlib.pyplot as plt


def plot_crime_by_year(df, year_column, value_column):
    """
    Plot total crime by year.
    """
    plt.figure()
    plt.plot(df[year_column], df[value_column], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Total Crime")
    plt.title("Total Crime by Year")
    plt.grid()
    plt.tight_layout()
    return plt


def plot_crime_by_county(df, county_column, value_column):
    """
    Plot total crime by county as a bar chart.
    """
    plt.figure()
    plt.bar(df[county_column], df[value_column])
    plt.xlabel("County")
    plt.ylabel("Total Crime")
    plt.title("Total Crime by County")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt


def plot_top_risk_areas(risk_df, value_column):
    """
    Plot top predicted crime areas.
    """
    plt.figure()
    plt.barh(risk_df.index.astype(str), risk_df[value_column])
    plt.xlabel("Predicted Crime")
    plt.title("Top High-Risk Areas")
    plt.tight_layout()
    return plt
