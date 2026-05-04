import pandas as pd

from bigdatatech_final_project.visualization import (
    plot_crime_by_year,
    plot_crime_by_county,
)


def test_plot_crime_by_year():
    df = pd.DataFrame({
        "year": [2020, 2021],
        "index_total": [100, 200],
    })

    plot = plot_crime_by_year(df, "year", "index_total")

    assert plot is not None


def test_plot_crime_by_county():
    df = pd.DataFrame({
        "county": ["A", "B"],
        "index_total": [100, 200],
    })

    plot = plot_crime_by_county(df, "county", "index_total")

    assert plot is not None
