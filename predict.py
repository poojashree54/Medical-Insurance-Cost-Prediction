# predict.py
from fastapi import FastAPI
import pickle

# Create FastAPI app
app = FastAPI()

# Load model and DictVectorizer
with open("model.pkl", "rb") as f_in:
    model = pickle.load(f_in)

with open("dv.pkl", "rb") as f_in:
    dv = pickle.load(f_in)

# Test route
@app.get("/ping")
def ping():
    return {"message": "✅ API is running successfully!"}

# Prediction route
@app.post("/predict")
def predict_insurance(data: dict):
    X = dv.transform([data])
    prediction = model.predict(X)[0]
    return {"predicted_charges": round(float(prediction), 2)}


# sample = {
#     "age": 35,
#     "sex": "male",
#     "bmi": 27.8,
#     "children": 2,
#     "smoker": "no",
#     "region": "southwest"
# }

# predict_insurance(sample)
