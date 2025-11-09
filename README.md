# Medical Insurance Cost Prediction

## Project Overview

This project predicts the **annual medical insurance charges** for individuals based on demographic and health-related factors such as **age, gender, BMI, number of children, smoking status, and region**.

It uses multiple **regression-based machine learning models** — Linear Regression, Decision Tree, Random Forest, and XGBoost — to estimate costs and deploys the best-performing model as a **FastAPI web service**, fully **containerized with Docker**.


##  Project Motivation

Healthcare costs can vary significantly depending on personal and lifestyle factors such as **age**, **BMI**, and **smoking habits**.

The goal of this project is to:

* Build a regression model to predict **insurance charges** based on individual attributes.
* Analyze which features most influence the cost (e.g., smoker, BMI, age).
* Deploy a **reproducible FastAPI web service** that returns predictions on demand.


## Dataset Information

**Source:** [Medical Cost Personal Dataset – Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)

| Feature      | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| **age**      | Age of the individual                                         |
| **sex**      | Gender (male/female)                                          |
| **bmi**      | Body Mass Index (weight/height²)                              |
| **children** | Number of dependents covered by insurance                     |
| **smoker**   | Smoking status (yes/no)                                       |
| **region**   | Residential area (northeast, northwest, southeast, southwest) |
| **charges**  | Annual medical insurance cost (Target variable)               |

**Target Variable:** `charges`


##  Tech Stack

* **Language:** Python 3.11
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
* **API Framework:** FastAPI
* **Containerization:** Docker
* **Version Control:** Git + GitHub


##  Exploratory Data Analysis (EDA)

* Analyzed data distributions and correlations
* Checked skewness in the `charges` target 
* Compared smokers vs non-smokers — strong cost difference observed
* Identified **smoker**, **bmi**, and **age** as top predictors
* Visualized patterns using boxplots, heatmaps, and pairplots


##  Modeling Approach

### Models Trained:

1. **Linear Regression** – Baseline
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **XGBoost Regressor** –  *Best performer*

### Preprocessing:

* One-Hot Encoding for categorical features (`sex`, `smoker`, `region`)
* Standard Scaling for numerical features (`age`, `bmi`, `children`)
* Data Split: 80% training / 20% testing

### Hyperparameter Tuning:

* Tuned key parameters like `max_depth`, `n_estimators`, and `learning_rate`


## Evaluation Metrics

| Metric       | Description                           | 
| ------------ | ------------------------------------- | 
| **RMSE**     | Root Mean Squared Error (main metric) |
| **MAE**      | Mean Absolute Error                   |
| **R² Score** | Coefficient of Determination          | 

---

## Results

| Model              | R² Score |
| ------------------ | -------- |
| Linear Regression  | 0.75     |
| Decision Tree      | 0.83     |
| Random Forest      | 0.87     |
| **XGBoost (Best)** | **0.88** |

**Top Predictors:**

* `smoker` → highest impact
* `bmi`, `age`, and `children` → secondary influence


## How to Run the Project

### Clone the repository:

```bash
git clone https://github.com/<your-username>/Medical-Insurance-Cost-Prediction.git
cd Medical-Insurance-Cost-Prediction
```

### Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Train the model:

```bash
python train.py
```

### Run API (local):

```bash
uvicorn predict:app --reload --port 8000
```

Visit your API docs → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Docker Deployment

### Build Docker image:

```bash
docker build -t insurance-api .
```

###  Run container:

```bash
docker run -p 8000:8000 insurance-api
```

### Access API:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)


##  Example API Usage

### Input JSON:

```json
{
    "age": 35,
    "sex": "male",
    "bmi": 27.8,
    "children": 2,
    "smoker": "no",
    "region": "southwest"
}
```

### Response:

```json
{
  "predicted_charges": 6218.95
}
```

### Acknowledgment
- Dataset from [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Inspired by the [DataTalks.Club ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
