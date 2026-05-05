# Bigdatatech_sp26_finalproject-pangweiliu

# NYC Crime Data Analysis & Predictive Modeling

## Project Overview
This project examines crime statistics in New York State through a
comprehensive data pipeline initiated in 1990, encompassing data cleansing,
aggregation, visualization, and machine learning modeling. The objective is to
examine historical crime patterns and enhance the analysis by forecasting crime
intensity across various geographies utilizing the scikit-learn tool. The origin of
the raw data (.csv) is from: 
https://data.ny.gov/Public-Safety/Index-Crimes-by-County-and-Agency-Beginning-1990/ca8h-8gjq/about_data.

## Project Structure

src/
|------bigdatatech_final_project/
       |--- data_loader.py
       |--- cleaning.py
       |--- analysis.py
       |--- visualization.py 
       |--- crime_model_demo.py # ML model for predicting crime rate

database/
|-------Index_Crimes_by_County_and_Agency__Beginning_1990.csv

main.py
README.md

## Research Methods
This project adheres to a systematic data analysis pipeline that integrates data
pretreatment, exploratory analysis, visualization, and machine learning. The raw
crime dataset was initially refined by standardizing column names, addressing
missing values, and assuring uniform data types. A descriptive study was
subsequently performed to investigate crime trends over time and across
counties utilizing aggregation procedures. Visualization methods, such as line
and bar charts, and scatter plot were employed to emphasize temporal and regional patterns. A
machine learning model was developed with scikit-learn to enhance the analysis. 
A Random Forest Regressor was employed to forecast total crime counts
(index_total) utilizing temporal (year) and categorical (county, agency) variables. 
The pipeline incorporated preprocessing techniques, including imputation and
one-hot encoding, to manage heterogeneous data types and guarantee model
compliance.

## How to run the analysis and model
   1) From the project root: PYTHONPATH=src python main.py
   2) Optional: run the standalone model script
        python src/bigdatatech_final_project/crime_model_demo.py \
          --data database/Index_Crimes_by_County_and_Agency__Beginning_1990.csv \
          --target index_total

## Results
The prediction model demonstrated robust performance, with a Mean Absolute
Error (MAE) of roughly 65.6, a Root Mean Squared Error (RMSE) of 109.2, and a R²
value of 0.725. This signifies that the model accounts for around 72.5% of the
variance in crime levels, indicating proficient pattern recognition within the data. 
The program revealed numerous high-risk locales, with Orange County State
Police continuously ranking among the top projected crime locations over
several years. Furthermore, the examination of feature importance indicated
that specific crime-related indicators and temporal variables significantly
influenced the predictions. The findings indicate that crime patterns
demonstrate regional grouping and temporal regularity.

## Limitations
Notwithstanding the robust performance of the model, certain limitations
warrant consideration. Initially, certain input properties, including specific crime
categories (e.g., robbery, rape), are intrinsically associated with the target
variable (index_total), thereby leading to target leakage and an overestimation of
predictive accuracy. The model fails to consider external socio-economic or
demographic variables that may affect crime patterns, such population density,
income levels, or law enforcement procedures. Third, crime data may exhibit
reporting bias, indicating that documented instances may not accurately reflect
the true incidence of crime. Consequently, the model need to be regarded as an
exploratory analytical instrument rather than a conclusive forecasting mechanism.

## Conclusions
This project illustrates how a systematic data pipeline integrated with machine
learning can yield insights into crime patterns and risk distribution. The
incorporation of scikit-learn augments analysis by facilitating predictive
functionalities and more profound data interpretation.

## Future Works
Future enhancements to this project may concentrate on augmenting data
quality and refining modeling methodologies. A primary objective is to enhance
feature selection by eliminating potentially confounding variables and integrating
independent predictors, including socio-economic indices and geographic data. 
Furthermore, more sophisticated machine learning models, such as gradient
boosting or time-series models, may be investigated to enhance predictive
efficacy. Integrating geospatial visualization (e.g., charting crime risk across
regions) might yield more intuitive findings. Ultimately, incorporating temporal
forecasting and scenario-based simulations could enhance decision-making in
resource allocation and crime prevention measures.
