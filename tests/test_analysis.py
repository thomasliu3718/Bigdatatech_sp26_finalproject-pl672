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
