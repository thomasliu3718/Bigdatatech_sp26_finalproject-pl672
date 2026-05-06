import pandas as pd

from bigdatatech_final_project.crime_model_demo import (
    clean_crime_data,
    build_features,
    train_crime_model,
    evaluate_model,
    make_area_risk_table,
    get_feature_importance,
)


def sample_crime_df():
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


def test_clean_crime_data_returns_dataframe():
    df = sample_crime_df()
    cleaned = clean_crime_data(df)

    assert not cleaned.empty
    assert "index_total" in cleaned.columns


def test_build_features_removes_target():
    df = sample_crime_df()
    cleaned = clean_crime_data(df)

    X, y, num_cols, cat_cols = build_features(cleaned, "index_total")

    assert "index_total" not in X.columns
    assert len(X) == len(y)
    assert "year" in num_cols
    assert "county" in cat_cols


def test_train_crime_model_predicts():
    df = clean_crime_data(sample_crime_df())
    X, y, num_cols, cat_cols = build_features(df, "index_total")

    model = train_crime_model(X, y, num_cols, cat_cols)
    preds = model.predict(X.head())

    assert len(preds) == len(X.head())


def test_evaluate_model_returns_metrics():
    df = clean_crime_data(sample_crime_df())
    X, y, num_cols, cat_cols = build_features(df, "index_total")

    model = train_crime_model(X, y, num_cols, cat_cols)
    metrics = evaluate_model(model, X, y)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics


def test_make_area_risk_table_returns_top_rows():
    df = clean_crime_data(sample_crime_df())
    X, y, num_cols, cat_cols = build_features(df, "index_total")

    model = train_crime_model(X, y, num_cols, cat_cols)
    risk_table = make_area_risk_table(
        model=model,
        X=X,
        original_df=df,
        target_col="index_total",
        top_n=5,
    )

    assert len(risk_table) == 5
    assert "predicted_index_total" in risk_table.columns


def test_get_feature_importance_returns_dataframe():
    df = clean_crime_data(sample_crime_df())
    X, y, num_cols, cat_cols = build_features(df, "index_total")

    model = train_crime_model(X, y, num_cols, cat_cols)
    importance = get_feature_importance(model)

    assert "feature" in importance.columns
    assert "importance" in importance.columns
    assert len(importance) > 0
