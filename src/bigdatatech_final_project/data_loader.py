"""
data_loader.py

Functions for loading the NYC/New York crime dataset.
"""

from pathlib import Path
import pandas as pd


def load_crime_data(file_path):
    """
    Load a crime dataset from a CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the crime CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded crime dataset.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    ValueError
        If the file is not a CSV file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("Input file must be a CSV file.")

    return pd.read_csv(file_path)
