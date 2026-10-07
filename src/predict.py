from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.data_cleaning import load_or_create_dataset

MODEL_PATHS = [
    ROOT / "models" / "churn_model.pkl",
]

REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "internet_service",
    "support",
]


def _coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_customer_payload(payload):
    record = {}
    alias_map = {
        "tenure": ["tenure"],
        "monthly_charges": ["monthly_charges", "MonthlyCharges"],
        "total_charges": ["total_charges", "TotalCharges"],
        "contract_type": ["contract_type", "Contract"],
        "internet_service": ["internet_service", "InternetService", "internet"],
        "support": ["support", "TechSupport"],
    }

    for canonical_name, aliases in alias_map.items():
        value = None
        for alias in aliases:
            if alias in payload:
                value = payload[alias]
                break
        if value is None:
            if canonical_name == "tenure":
                value = 12
            elif canonical_name == "monthly_charges":
                value = 70.0
            elif canonical_name == "total_charges":
                value = 840.0
            elif canonical_name == "contract_type":
                value = "Month-to-month"
            elif canonical_name == "internet_service":
                value = "Fiber"
            elif canonical_name == "support":
                value = "No"

        if canonical_name in {"tenure", "monthly_charges", "total_charges"}:
            record[canonical_name] = _coerce_float(value, 0)
        else:
            record[canonical_name] = value

    return pd.DataFrame([record], columns=REQUIRED_COLUMNS)


def predict_churn(model, payload):
    """Return a 0/1 churn prediction for a single record."""
    customer_df = _normalize_customer_payload(payload)
    return int(model.predict(customer_df)[0])


def _resolve_model_path():
    for path in MODEL_PATHS:
        if path.exists():
            return path

    from src.train import train_and_save_model

    dataset = load_or_create_dataset()
    train_and_save_model(dataset)
    return ROOT / "models" / "churn_model.joblib"


def main():
    model_path = _resolve_model_path()
    model = joblib.load(model_path)

    customer = {
        "tenure": 5,
        "monthly_charges": 85.5,
        "total_charges": 427.5,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber",
        "support": "No",
    }

    customer_df = _normalize_customer_payload(customer)
    prediction = int(model.predict(customer_df)[0])
    probability = model.predict_proba(customer_df)[0][1]

    result = (
        "Customer is likely to churn"
        if prediction == 1
        else "Customer is unlikely to churn"
    )

    print(result)
    print(f"Churn probability: {probability:.2%}")


if __name__ == "__main__":
    main()