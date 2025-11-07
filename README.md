# Medical Insurance Cost Prediction

### Project Overview

This project predicts the annual medical insurance charges for individuals based on demographic and health-related attributes such as age, gender, BMI, number of children, smoking status, and region.

It uses regression-based machine learning models — Linear Regression, Decision Tree, Random Forest, and XGBoost — to estimate costs and deploys the final model as a FastAPI web service containerized with Docker.

### Table of Contents
- Project Motivation
- Dataset Information
- Tech Stack
- Exploratory Data Analysis (EDA)
- Modeling Approach
- Evaluation Metrics
- Results

### Project Motivation

Healthcare costs can vary significantly depending on personal factors such as lifestyle, age, and medical history.

This project aims to:
- Build a regression model to predict insurance charges based on given attributes.
- Analyze which features most affect the cost (e.g., smoking, BMI, age).
- Deploy a reproducible web API that serves predictions on demand.

### Dataset Information

Dataset: Medical Cost Personal Dataset – Kaggle

Feature	Description
- age	Age of the individual
- sex	Gender (male/female)
- bmi	Body Mass Index (weight/height²)
- children	Number of dependents covered by insurance
- smoker	Smoking status (yes/no)
- region	Residential area (northeast, northwest, southeast, southwest)
- charges	Medical insurance cost (Target variable)

Target Variable: charges (annual insurance cost)

### Tech Stack

- Language: Python 3.11
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, joblib
- API Framework: FastAPI
- Containerization: Docker
- Version Control: Git & GitHub

### Exploratory Data Analysis (EDA)

- Performed in notebooks/01_eda.ipynb:
- Analyzed data distributions and correlations
- Identified skewness in the charges target (applied log1p transformation)
- Compared smokers vs non-smokers
- Found smoker, bmi, and age as the top cost-driving factors
- Visualized key patterns with boxplots and heatmaps

### Modeling Approach

Trained and compared the following regression models:
- Linear Regression (Baseline)
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor (Best performer)

Feature Preprocessing:
- One-Hot Encoding for categorical features (sex, smoker, region)
- Standard Scaling for numerical features (age, bmi, children)

Data split: 80% train / 20% test

Hyperparameter Tuning:
- GridSearchCV / RandomizedSearchCV on RandomForest and XGBoost

### Evaluation Metrics

Metric	Description
- RMSE	Root Mean Squared Error – main evaluation metric
- MAE	Mean Absolute Error
- R² Score	Coefficient of Determination

### Results

Top features impacting insurance cost: smoker, bmi, age, and children.

Best model: XGBoost Regressor.
