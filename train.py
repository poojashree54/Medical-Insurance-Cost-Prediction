# train.py
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("insurance.csv")

# Split data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

y_train = df_full_train.charges.values
y_test = df_test.charges.values
X_train = df_full_train.drop('charges', axis=1)
X_test = df_test.drop('charges', axis=1)

# Feature setup
cat = ['sex', 'smoker', 'region']
num = ['age', 'bmi', 'children']

dv = DictVectorizer(sparse=False)

train_dict = X_train[cat + num].to_dict(orient='records')
X_train_encoded = dv.fit_transform(train_dict)

# Train XGBoost model
model = XGBRegressor(
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train_encoded, y_train)

# Evaluate performance
val_dict = X_test[cat + num].to_dict(orient='records')
X_val_encoded = dv.transform(val_dict)
y_pred = model.predict(X_val_encoded)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation R²: {r2:.4f}")

# Save model and DictVectorizer
with open("model.pkl", "wb") as f_out:
    pickle.dump(model, f_out)

with open("dv.pkl", "wb") as f_out:
    pickle.dump(dv, f_out)

print("Model and DictVectorizer saved successfully!")