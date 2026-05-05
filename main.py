"""
main.py

Run the full crime data analysis pipeline.
"""
import pandas as pd

from bigdatatech_final_project.visualization import (
    plot_crime_by_year,
    plot_crime_by_county,
    plot_top_risk_areas,
)

from bigdatatech_final_project.data_loader import load_crime_data
from bigdatatech_final_project.cleaning import (
    standardize_column_names,
    remove_missing_rows,
)
from bigdatatech_final_project.analysis import (
    total_crime_by_year,
    total_crime_by_county,
    average_crime,
    crime_model_analysis,
    
)
def map_columns(df):
    """
    Map real NYC dataset columns to:
    year, county, index_total
    Adjust the mapping to match your dataset.
    """
    # Example mappings (edit to match your columns)
    mapping = {
        "year": "year",                      # if already 'year'
        "report_year": "year",
        "cmplnt_fr_dt": "year",             # if you later extract year
        "borough": "county",
        "county": "county",
        "index_total": "index_total",
        "complaint_count": "index_total",
        "count": "index_total",
    }

    # keep only columns that exist in df
    mapping = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=mapping)


def main():
    # Step 1: Load data
    file_path = "database/Index_Crimes_by_County_and_Agency__Beginning_1990_20260504.csv"  # change to your actual file
    df = load_crime_data(file_path)

    # Step 2: Clean data
    df = standardize_column_names(df)
    df = remove_missing_rows(df)
    df = map_columns(df)
    
    df["index_total"] = pd.to_numeric(df["index_total"], errors="coerce")
    df = remove_missing_rows(df)


    # Step 3A: Common Analyze
    yearly = total_crime_by_year(df, "year", "index_total")
    county = total_crime_by_county(df, "county", "index_total")
    avg = average_crime(df, "index_total")

    # Step 3B: Machine Learning Analysis
    metrics, risk_table = crime_model_analysis(df, "index_total")

    print("\n=== Crime Model Metrics ===")
    print(metrics)
    print("\n=== Top Risk Areas ===")
    print(risk_table)
    

    # Step 4: Output
    print("\n=== Total Crime by Year ===")
    print(yearly)

    print("\n=== Total Crime by County ===")
    print(county)

    print("\n=== Average Crime ===")
    print(avg)

    # Step 5: Visualization
    plot_crime_by_year(yearly, "year", "index_total")
    plot_crime_by_county(county, "county", "index_total")
    plot_top_risk_areas(risk_table, "predicted_index_total")
    
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    main()
