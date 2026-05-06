# NYC Crime Data Analysis & Predictive Modeling

## Project Overview
This project examines crime statistics in New York State through a
comprehensive data pipeline initiated in 1990 to 2024, encompassing data cleansing,
aggregation, visualization, and machine learning modeling. The objective is to
examine historical crime patterns and enhance the analysis by forecasting crime
intensity across various geographies utilizing the scikit-learn tool. The origin of
the raw data (.csv) is from: 
https://data.ny.gov/Public-Safety/Index-Crimes-by-County-and-Agency-Beginning-1990/ca8h-8gjq/about_data.

## Project Structure
```text

Bigdatatech_sp26_finalproject-pl672/
|
|---- doc/
|
|
|---- src/
|	     |---- bigdatatech_final_project/       		
|		                         	    |--- __init__.py
|		                         	    |--- data_loader.py
|      		                            |--- cleaning.py
|      		                            |--- analysis.py
|       	                            |--- visualization.py 
|       	                            |--- crime_model_demo.py # ML model for predicting crime rate
|	
|--- database/
|	         |--- Index_Crimes_by_County_and_Agency__Beginning_1990.csv
|
|--- tests/
|
|--- main.py
|--- README.md
|--- pyproject.toml
|--- LICENSE.txt

```

## Research Methods
This project implements a comprehensive data analytics workflow to analyze crime patterns throughout New
York State utilizing Python and machine learning methodologies. The dataset underwent initial cleaning and
standardization via preprocessing methods such as column normalization, management of missing values, and
aggregation of county-level crime statistics. An exploratory data analysis was subsequently performed to assess
historical crime trends over the years and across counties through statistical summaries and visualization methods.

A predictive criminal modeling framework was established utilizing the scikit-learn package to surpass
descriptive analytics. A Random Forest machine learning pipeline was developed to predict total crime counts
(index_total) utilizing temporal, geographic, and crime-category variables. Feature importance analysis was
conducted to ascertain the variables that most significantly influence prediction performance. A trend-based
forecasting method was employed to project future crime counts for the Orange County State Police in 2025
and 2026. Various visualization techniques, such as ranking risk diagrams and predictive charts, were created 
to enhance interpretability and presentation quality.

## How to run the analysis and the predictive model
```text
   
   1) From the project root: PYTHONPATH=src python main.py
   2) Optional: run the standalone model script
        python src/bigdatatech_final_project/crime_model_demo.py \
          --data database/Index_Crimes_by_County_and_Agency__Beginning_1990.csv \
          --target index_total

```

## Results
The study effectively discerned significant temporal and geographic crime patterns within the dataset. Historical
data revealed an overall decrease in total crime throughout New York State from 2002 to 2024, however several
counties continually displayed high crime rates. Westchester County, Erie County, and Orange County are among 
the counties with the highest total crime rates.

The predictive machine learning model demonstrated robust performance, yielding a R² value of roughly 0.73,
signifying that the model accounted for a considerable percentage of the variance in crime counts. Feature
importance analysis indicated that variables associated with robbery, variables connected to rape, and temporal
indicators were among the most significant predictors in the model.

The forecasting extension projected future crime counts for the Orange County State Police, estimating around
480 events in 2025 and 450 incidents in 2026. The visualization of these projections in conjunction with
historical data indicated a minor resurgence in criminal activity after the reduction observed post-2020,
succeeded by a phase of moderate stabilization. The predictive visualization system effectively differentiated
historical observations from expected values using distinct color coding and comparative graphing.

## Limitations
Numerous constraints must be recognized when analyzing the project outcomes. The predictive model solely
depends on characteristics within the crime dataset and excludes external socio-economic, demographic, or
environmental factors that could substantially affect crime trends. Secondly, certain crime-category variables
are closely associated with the target variable (index_total), potentially inflating prediction accuracy due to 
information overlap.

The future forecasting component utilizes past trends and recent patterns to estimate future values; thus, 
the predictions should be regarded as exploratory forecasts rather than exact real-world projections. 
Inconsistencies in crime reporting, absence of contextual information, and potential reporting bias in the 
original dataset may potentially impact model accuracy. The analysis was conducted at the county and agency
level, potentially overlooking particular neighborhood crime dynamics.

## Conclusions
This project illustrates the integration of big data analytics and machine learning methodologies to analyze and
predict crime patterns utilizing publicly accessible crime records. The integration of preprocessing, visualization,
predictive modeling, and trend forecasting yielded both descriptive and predictive insights into regional crime patterns 
throughout New York State.

The findings indicate that machine learning methodologies can proficiently identify temporal and spatial crime
patterns, while also facilitating future predictive applications. The project underscores the significance of
visualization design in effectively and professionally conveying analytical results. The research creates a scalable
analytical framework that may be expanded to facilitate future crime monitoring and decision-making applications.

## Future Works
Future enhancements may broaden the analytical scope and elevate the prediction sophistication of the project. 
Incorporating supplementary external variables, including population density, unemployment rates, economic
indicators, or geographic information system (GIS) data, could enhance the model's realism and predictive
capability. Advanced forecasting methodologies, such as gradient boosting, XGBoost, recurrent neural networks, 
or time-series models, may significantly improve long-term prediction accuracy.

Future endeavors may potentially integrate interactive dashboards and GIS heatmaps to facilitate more natural 
depiction of regional crime risk. Furthermore, expanding the framework to forecast distinct crime types
separately, rather than solely aggregate crime totals, may facilitate more precise analytical findings. 
The existing forecasting framework could be modified into a real-time monitoring system for public safety 
analytics and resource allocation applications.
