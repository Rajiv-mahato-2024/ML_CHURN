from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
PREDICTIONS_PATH = ROOT / "data" / "monitoring" / "predictions.jsonl"
ERRORS_PATH = ROOT / "data" / "monitoring" / "errors.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Churn Prediction API", version="2.0.0")
model = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
registry_manifest_path = ROOT / "models" / "registry" / "latest.json"
if registry_manifest_path.is_file():
    with registry_manifest_path.open(encoding="utf-8") as manifest_file:
        model_version = json.load(manifest_file)["version"]
else:
    model_version = "unregistered"


class Customer(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    tenure: int = Field(ge=0)
    monthly_charges: float | None = Field(default=None, alias="MonthlyCharges")
    total_charges: float | None = Field(default=None, alias="TotalCharges")
    contract_type: str | None = Field(default=None, alias="Contract")
    internet_service: str | None = Field(default=None, alias="InternetService")
    support: str | None = Field(default=None, alias="TechSupport")


def _record_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event, allow_nan=False) + "\n")


def _model_input(customer: Customer) -> dict[str, Any]:
    values = customer.model_dump()
    monthly_charges = values["monthly_charges"]
    total_charges = values["total_charges"]
    contract_type = values["contract_type"]
    internet_service = values["internet_service"]
    support = values["support"]
    if any(
        value is None
        for value in (
            monthly_charges,
            total_charges,
            contract_type,
            internet_service,
            support,
        )
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide monthly_charges, total_charges, contract_type, "
                "internet_service, and support (or their Telco field aliases)."
            ),
        )

    internet_service = {
        "Fiber optic": "Fiber",
        "No": "None",
        "No internet service": "None",
    }.get(internet_service, internet_service)
    support = {
        "No internet service": "No",
    }.get(support, support)
    return {
        "tenure": values["tenure"],
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "support": support,
    }


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Customer Churn Prediction API", "version": app.version}


@app.get("/health")
def health() -> dict[str, str]:
    if model is None:
        return {"status": "unhealthy"}
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict")
def predict(customer: Customer) -> dict[str, int | float]:
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction model is unavailable.")

    logger.info("Prediction request received")
    try:
        model_input = _model_input(customer)
        customer_df = pd.DataFrame([model_input])
        prediction = int(model.predict(customer_df)[0])
        probability = float(model.predict_proba(customer_df)[0][1])
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            **model_input,
            "prediction": prediction,
            "churn_probability": probability,
            "model_version": model_version,
        }
        _record_event(PREDICTIONS_PATH, event)
        logger.info("Prediction completed: %s", prediction)
        return {"prediction": prediction, "churn_probability": probability}
    except HTTPException:
        raise
    except Exception as error:
        _record_event(
            ERRORS_PATH,
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
            },
        )
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from error
