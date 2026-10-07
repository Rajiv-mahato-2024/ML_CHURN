from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "customer_churn.csv"


def make_sample_dataframe(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_ids = [f"C{i:05d}" for i in range(1, n + 1)]
    tenure = rng.integers(1, 72, size=n)
    monthly_charges = rng.uniform(30.0, 120.0, size=n)
    total_charges = monthly_charges * rng.uniform(4.0, 18.0, size=n) + tenure * 3.5
    contract_types = np.array(["Month-to-month", "One year", "Two year"])
    internet_services = np.array(["DSL", "Fiber", "None"])
    support = np.array(["No", "Yes"])

    contract = contract_types[rng.integers(0, len(contract_types), size=n)]
    internet = internet_services[rng.integers(0, len(internet_services), size=n)]
    support_flags = support[rng.integers(0, len(support), size=n)]

    churn_probability = (
        0.12
        + (contract == "Month-to-month") * 0.22
        + (internet == "Fiber") * 0.18
        + (support_flags == "Yes") * 0.05
        + (tenure < 12) * 0.20
    )
    churn = (rng.random(n) < churn_probability).astype(int)

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract,
            "internet_service": internet,
            "support": support_flags,
            "churn": churn,
        }
    )


def load_data(path):
    return pd.read_csv(path)


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce").fillna(1).clip(lower=1)
    cleaned["monthly_charges"] = pd.to_numeric(cleaned["monthly_charges"], errors="coerce")
    cleaned["total_charges"] = pd.to_numeric(cleaned["total_charges"], errors="coerce")

    cleaned["churn"] = cleaned["churn"].map({0: 0, 1: 1, "No": 0, "Yes": 1})
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned[cleaned["monthly_charges"] > 0].copy()

    if cleaned["total_charges"].notna().any():
        cleaned["total_charges"] = cleaned["total_charges"].fillna(
            cleaned["total_charges"].median()
        )

    return cleaned


def clean_data(df):
    return clean_customer_data(df)


def load_or_create_dataset(path: str | Path = DATA_PATH):
    path = Path(path)

    if path.exists():
        return clean_customer_data(load_data(path))

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_df = make_sample_dataframe()
    sample_df.to_csv(path, index=False)
    return clean_customer_data(sample_df)