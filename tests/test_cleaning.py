import pandas as pd

from bigdatatech_final_project.cleaning import (
    standardize_column_names,
    remove_missing_rows,
    filter_by_year_range,
)


def test_standardize_column_names():
    df = pd.DataFrame({
        " County Name ": ["New York"],
        "Index Total": [1000],
        "Violent Crime": [200],
    })

    result = standardize_column_names(df)

    assert list(result.columns) == [
        "county_name",
        "index_total",
        "violent_crime",
    ]


def test_remove_missing_rows():
    df = pd.DataFrame({
        "county": ["New York", None, "Kings"],
        "year": [2020, 2021, 2022],
        "index_total": [1000, 1200, None],
    })

    result = remove_missing_rows(df)

    assert result.shape[0] == 1
    assert result.loc[0, "county"] == "New York"


def test_filter_by_year_range():
    df = pd.DataFrame({
        "county": ["A", "B", "C", "D"],
        "year": [2018, 2019, 2020, 2021],
        "index_total": [100, 200, 300, 400],
    })

    result = filter_by_year_range(df, "year", 2019, 2020)

    assert result.shape[0] == 2
    assert list(result["year"]) == [2019, 2020]
