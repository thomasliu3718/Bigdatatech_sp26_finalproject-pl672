"""
crime_model_demo.py

Optional extension analysis for the Big Data Technology final project.

Goal
----
Use scikit-learn to demonstrate how historical crime data can be used to
predict the potential risk/count of a selected crime type in assigned areas.

This script is intentionally separate from credit_shift.py so it will not
interfere with autograder-required functions.

Expected input
--------------
A CSV file with at least some of the following columns:
- County
- Agency
- Year
- Index Total
- Murder
- Rape
- Robbery
- Aggravated Assault
- Burglary
- Larceny
- Motor Vehicle Theft

Example usage
-------------
python crime_model_demo.py --data index_crimes.csv --target "Burglary"

If your target column is numeric, the script trains a regression model to
predict the crime count for that target.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_TARGET = "Burglary"


def load_crime_data(path: str) -> pd.DataFrame:
    """Load crime CSV data."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_crime_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for crime modeling."""
    cleaned = df.copy()

    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    for col in cleaned.columns:
        if col not in cleaned.select_dtypes(include=["object"]).columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if "Year" in cleaned.columns:
        cleaned["Year"] = pd.to_numeric(cleaned["Year"], errors="coerce")
        cleaned["years_since_start"] = cleaned["Year"] - cleaned["Year"].min()

    return cleaned


def build_features(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Create X/y and identify numeric/categorical columns."""
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found. "
            f"Available columns are: {list(df.columns)}"
        )

    y = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = y.notna()

    data = df.loc[valid_mask].copy()
    y = y.loc[valid_mask]

    leakage_cols = {target_col}
    for c in data.columns:
        c_lower = c.lower()
        if "total" in c_lower and target_col.lower() not in c_lower:
            leakage_cols.add(c)

    X = data.drop(columns=[c for c in leakage_cols if c in data.columns])

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    return X, y, numeric_cols, categorical_cols


def train_crime_model(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Pipeline:
    """Train a Random Forest regression model for crime count prediction."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X, y)
    return model


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate regression performance."""
    pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, pred) ** 0.5

    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_test, pred)),
    }


def make_area_risk_table(
    model: Pipeline,
    X: pd.DataFrame,
    original_df: pd.DataFrame,
    target_col: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Create a ranked table of areas with highest predicted crime counts."""
    pred = model.predict(X)

    result = original_df.loc[X.index].copy()
    result[f"predicted_{target_col}"] = pred

    display_cols = []
    for col in ["county", "agency", "year"]:
        if col in result.columns:
            display_cols.append(col)

    display_cols.append(f"predicted_{target_col}")

    return (
        result[display_cols]
        .sort_values(f"predicted_{target_col}", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to crime CSV file")
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Crime type/count column to predict. Default: {DEFAULT_TARGET}",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of highest-risk area rows to display",
    )
    args = parser.parse_args()

    df = load_crime_data(args.data)
    df = clean_crime_data(df)

    X, y, numeric_cols, categorical_cols = build_features(df, args.target)

    if len(X) < 20:
        raise ValueError("Not enough rows for train/test modeling after cleaning.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = train_crime_model(X_train, y_train, numeric_cols, categorical_cols)
    metrics = evaluate_model(model, X_test, y_test)

    print("\nCrime Model Demo")
    print("================")
    print(f"Target crime column: {args.target}")
    print(f"Rows used: {len(X)}")
    print("\nModel evaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    risk_table = make_area_risk_table(
        model=model,
        X=X,
        original_df=df,
        target_col=args.target,
        top_n=args.top_n,
    )

    print(f"\nTop {args.top_n} predicted high-risk area/year rows:")
    print(risk_table.to_string(index=False))


if __name__ == "__main__":
    main()
    

def get_feature_importance(model):
    """
    Return feature importance from the trained pipeline.
    Handles one-hot encoded categorical features.
    """
    preprocessor = model.named_steps["preprocess"]
    regressor = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)


def predict_future_crime(model, template_df, county, agency, year=2026):
    """
    Predict future crime count for a selected county/agency/year.
    Uses median/mode values from existing data as defaults.
    """
    future_row = {}

    for col in template_df.columns:
        if col == "county":
            future_row[col] = county
        elif col == "agency":
            future_row[col] = agency
        elif col == "year":
            future_row[col] = year
        elif pd.api.types.is_numeric_dtype(template_df[col]):
            future_row[col] = template_df[col].median()
        else:
            future_row[col] = template_df[col].mode()[0]

    future_df = pd.DataFrame([future_row])

    pred = model.predict(future_df)[0]

    return future_df, float(pred)
