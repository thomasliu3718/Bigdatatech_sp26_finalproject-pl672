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


import matplotlib.pyplot as plt

from bigdatatech_final_project.visualization import (
    plot_crime_by_year,
    plot_crime_by_county,
    plot_top_risk_areas,
    plot_future_prediction,
)


def test_plot_crime_by_year_runs():
    df = pd.DataFrame({
        "year": [2020, 2021, 2022],
        "index_total": [100, 120, 90],
    })

    plot_crime_by_year(df, "year", "index_total")
    plt.close("all")


def test_plot_crime_by_county_runs():
    df = pd.DataFrame({
        "county": ["A", "B", "C"],
        "index_total": [100, 200, 150],
    })

    plot_crime_by_county(df, "county", "index_total")
    plt.close("all")


def test_plot_top_risk_areas_runs():
    risk_df = pd.DataFrame({
        "county": ["Orange", "Clinton"],
        "agency": ["Orange County State Police", "Clinton County State Police"],
        "year": [2025, 2026],
        "predicted_index_total": [480.0, 450.0],
    })

    plot_top_risk_areas(risk_df, "predicted_index_total")
    plt.close("all")


def test_plot_future_prediction_runs():
    history_df = pd.DataFrame({
        "year": [2022, 2023, 2024],
        "index_total": [470, 490, 430],
    })

    plot_future_prediction(
        history_df=history_df,
        year_column="year",
        value_column="index_total",
        future_predictions={2025: 480, 2026: 450},
        title="Test Future Prediction",
    )

    plt.close("all")
