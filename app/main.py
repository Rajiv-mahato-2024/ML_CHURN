from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = ROOT / "models" / "churn_model.pkl"

app = FastAPI(title="Customer Churn Prediction API")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. Run `python -m src.train` first."
    )

model = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict")
def predict(customer: Customer):
    customer_data = customer.model_dump()
    internet_service = {
        "Fiber optic": "Fiber",
        "No": "None",
    }.get(customer_data["InternetService"], customer_data["InternetService"])
    support = "Yes" if customer_data["TechSupport"] == "Yes" else "No"

    customer_df = pd.DataFrame(
        [
            {
                "tenure": customer_data["tenure"],
                "monthly_charges": customer_data["MonthlyCharges"],
                "total_charges": customer_data["TotalCharges"],
                "contract_type": customer_data["Contract"],
                "internet_service": internet_service,
                "support": support,
            }
        ]
    )
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
    }
