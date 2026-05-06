import pandas as pd

from bigdatatech_final_project.analysis import (
    total_crime_by_year,
    total_crime_by_county,
    average_crime,
)


def test_total_crime_by_year():
    df = pd.DataFrame({
        "year": [2020, 2020, 2021],
        "index_total": [100, 200, 300],
    })

    result = total_crime_by_year(df, "year", "index_total")

    assert result.loc[result["year"] == 2020, "index_total"].values[0] == 300
    assert result.loc[result["year"] == 2021, "index_total"].values[0] == 300


def test_total_crime_by_county():
    df = pd.DataFrame({
        "county": ["A", "A", "B"],
        "index_total": [100, 200, 300],
    })

    result = total_crime_by_county(df, "county", "index_total")

    assert result.loc[result["county"] == "A", "index_total"].values[0] == 300
    assert result.loc[result["county"] == "B", "index_total"].values[0] == 300


def test_average_crime():
    df = pd.DataFrame({
        "index_total": [100, 200, 300],
    })

    result = average_crime(df, "index_total")

    assert result == 200


from bigdatatech_final_project.analysis import (
    total_crime_by_year,
    total_crime_by_county,
    average_crime,
    crime_model_analysis,
)


def sample_analysis_df():
    return pd.DataFrame({
        "county": ["Orange", "Orange", "Clinton", "Clinton", "Albany", "Albany"] * 5,
        "agency": ["Agency A", "Agency B", "Agency A", "Agency B", "Agency A", "Agency B"] * 5,
        "year": list(range(2000, 2030)),
        "murder": [1, 2, 1, 3, 2, 1] * 5,
        "rape": [2, 1, 3, 2, 1, 2] * 5,
        "robbery": [5, 6, 4, 7, 3, 5] * 5,
        "burglary": [20, 25, 15, 30, 10, 18] * 5,
        "index_total": [100, 120, 90, 150, 80, 110] * 5,
    })


def test_total_crime_by_year_values():
    df = pd.DataFrame({
        "year": [2020, 2020, 2021],
        "index_total": [10, 20, 30],
    })

    result = total_crime_by_year(df, "year", "index_total")

    assert result.loc[result["year"] == 2020, "index_total"].iloc[0] == 30
    assert result.loc[result["year"] == 2021, "index_total"].iloc[0] == 30


def test_total_crime_by_county_values():
    df = pd.DataFrame({
        "county": ["A", "A", "B"],
        "index_total": [10, 15, 20],
    })

    result = total_crime_by_county(df, "county", "index_total")

    assert result.loc[result["county"] == "A", "index_total"].iloc[0] == 25
    assert result.loc[result["county"] == "B", "index_total"].iloc[0] == 20


def test_average_crime_value():
    df = pd.DataFrame({
        "index_total": [10, 20, 30],
    })

    assert average_crime(df, "index_total") == 20


def test_crime_model_analysis_returns_outputs():
    df = sample_analysis_df()

    metrics, risk_table, model, X = crime_model_analysis(df, "index_total")

    assert isinstance(metrics, dict)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert not risk_table.empty
    assert "predicted_index_total" in risk_table.columns
    assert model is not None
    assert not X.empty
