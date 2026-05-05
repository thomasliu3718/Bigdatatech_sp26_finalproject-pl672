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


def plot_crime_by_county(df, county_column, value_column, top_n=15):
    """
    Plot top N counties by total crime as a clean horizontal bar chart.
    """
    plot_df = (
        df.sort_values(value_column, ascending=False)
        .head(top_n)
        .sort_values(value_column, ascending=True)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df[county_column], plot_df[value_column])

    plt.xlabel("Total Crime")
    plt.ylabel("County")
    plt.title(f"Top {top_n} Counties by Total Crime")

    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    return plt


def plot_top_risk_areas(risk_df, value_column):
    """
    Plot top predicted high-risk areas as a ranked dot plot.
    """
    plot_df = risk_df.copy()
    plot_df["label"] = (
        plot_df["county"] + " | " +
        plot_df["agency"] + " | " +
        plot_df["year"].astype(str)
    )

    plot_df = plot_df.sort_values(value_column, ascending=True)

    plt.figure(figsize=(11, 6))
    plt.scatter(plot_df[value_column], plot_df["label"], s=80)

    plt.xlabel("Predicted Total Crime (index_total)")
    plt.ylabel("County | Agency | Year")
    plt.title("Ranked Predicted Crime Risk by Area")

    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    return plt
