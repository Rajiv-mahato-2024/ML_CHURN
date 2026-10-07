from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_cleaning import clean_customer_data, load_or_create_dataset
from src.feature_engineering import build_feature_pipeline

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.pkl"


def train_model(data):
    cleaned = clean_customer_data(data)
    if "customer_id" in cleaned.columns:
        X = cleaned.drop(columns=["customer_id", "churn"])
    else:
        X = cleaned.drop(columns=["churn"])
    y = cleaned["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_feature_pipeline()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }
    return model, metrics


def train_and_save_model(data, path: str | Path = MODEL_PATH):
    model, metrics = train_model(data)
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, metrics


if __name__ == "__main__":
    dataset = load_or_create_dataset()
    model, metrics = train_and_save_model(dataset)
    print(f"Training complete. Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 score: {metrics['f1']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
    sample_row = dataset.drop(columns=["churn"]).iloc[0].to_dict()
    print(f"Sample prediction: {model.predict(pd.DataFrame([sample_row]))[0]}")