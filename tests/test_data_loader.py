import pandas as pd
import pytest

from bigdatatech_final_project.data_loader import load_crime_data


def test_load_crime_data_success(tmp_path):
    test_file = tmp_path / "crime.csv"

    sample_data = pd.DataFrame({
        "County": ["New York", "Kings"],
        "Year": [2020, 2021],
        "Index Total": [1000, 1200]
    })


def test_load_crime_data_success(tmp_path):
    test_file = tmp_path / "crime.csv"

    sample_data = pd.DataFrame({
        "County": ["New York", "Kings"],
        "Year": [2020, 2021],
        "Index Total": [1000, 1200]
    })

    sample_data.to_csv(test_file, index=False)

    result = load_crime_data(test_file)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 3)
    assert list(result.columns) == ["County", "Year", "Index Total"]


def test_load_crime_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_crime_data("missing_file.csv")


def test_load_crime_data_not_csv(tmp_path):
    test_file = tmp_path / "crime.txt"
    test_file.write_text("This is not a CSV file.")

    with pytest.raises(ValueError):
        load_crime_data(test_file)
