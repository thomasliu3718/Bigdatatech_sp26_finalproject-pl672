"""
analysis.py

Functions for analyzing crime datasets.
"""

import pandas as pd
from bigdatatech_final_project.crime_model_demo import get_feature_importance
from bigdatatech_final_project.crime_model_demo import predict_future_crime

def total_crime_by_year(df, year_column, value_column):
    """
    Compute total crime per year.
    """
    return (
        df.groupby(year_column)[value_column]
        .sum()
        .reset_index()
    )


def total_crime_by_county(df, county_column, value_column):
    """
    Compute total crime per county.
    """
    return (
        df.groupby(county_column)[value_column]
        .sum()
        .reset_index()
    )


def average_crime(df, value_column):
    """
    Compute average crime value.
    """
    return df[value_column].mean()


from bigdatatech_final_project.crime_model_demo import (
    build_features,
    train_crime_model,
    evaluate_model,
    make_area_risk_table
)


def crime_model_analysis(df, target_col="index_total"):
    """
    Run ML model on crime data and return metrics + risk table.
    """
    X, y, num_cols, cat_cols = build_features(df, target_col)

    if len(X) < 20:
        return {"error": "Not enough data for modeling"}, None

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = train_crime_model(X_train, y_train, num_cols, cat_cols)
    metrics = evaluate_model(model, X_test, y_test)

    importance = get_feature_importance(model)
    print("\n=== Feature Importance ===")
    print(importance.head(10))

    risk_table = make_area_risk_table(
        model=model,
        X=X,
        original_df=df,
        target_col=target_col,
        top_n=10
    )

    future_df, future_prediction = predict_future_crime(
    model=model,
    template_df=X,
    county="New York",
    agency="New York City Police Department",
    year=2026
    )

    print("\n=== Predicted NYC Crime in 2026 ===")
    print(future_df)
    print("Predicted index_total:", future_prediction)

    return metrics, risk_table, model, X


