from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_cleaning import DATA_PATH, clean_customer_data, load_data
from src.feature_engineering import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.train import MODEL_PATH, train_model

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
REGISTRY_DIR = ROOT / "models" / "registry"
METRICS = ("accuracy", "f1", "roc_auc")
FeatureTarget = tuple[pd.DataFrame, pd.Series]


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_eda_report(data: pd.DataFrame) -> dict[str, Any]:
    churn = data["churn"].astype(int)
    numeric_summary = data[NUMERIC_COLUMNS].describe().to_dict()
    category_counts = {
        column: data[column].value_counts(dropna=False).to_dict()
        for column in CATEGORICAL_COLUMNS
    }
    return _json_value(
        {
            "rows": len(data),
            "columns": list(data.columns),
            "missing_values": data.isna().sum().to_dict(),
            "numeric_summary": numeric_summary,
            "category_counts": category_counts,
            "churn_rate": float(churn.mean()),
            "churn_counts": churn.value_counts().to_dict(),
        }
    )


def _get_features_and_target(
    data: pd.DataFrame,
) -> FeatureTarget:
    required = {*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, "churn"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            f"Training data is missing required columns: {missing}"
        )
    if data["churn"].isna().any():
        raise ValueError("Training labels must not contain missing churn values.")
    labels = set(data["churn"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Churn labels must be binary (0 or 1); found: {labels}"
        )

    X = data.drop(columns=["churn", "customer_id"], errors="ignore")
    y = data["churn"].astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Training data must contain at least two examples of each class."
        )
    return X, y


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def is_better_model(
    candidate_metrics: dict[str, Any], champion_metrics: dict[str, Any]
) -> bool:
    return all(
        candidate_metrics[metric] > champion_metrics[metric]
        for metric in METRICS
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(_json_value(payload), temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _evaluate_on_holdout(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    return evaluate_model(model, X_test, y_test)


def _promote_model(
    model: Any,
    metrics: dict[str, Any],
    model_path: Path,
    registry_dir: Path,
    *,
    reason: str,
) -> str:
    version = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    registry_dir.mkdir(parents=True, exist_ok=True)
    versioned_model_path = registry_dir / f"churn_model-{version}.pkl"
    joblib.dump(model, versioned_model_path)

    manifest = {
        "version": version,
        "registered_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "metrics": metrics,
        "model_file": versioned_model_path.name,
    }
    _write_json_atomic(registry_dir / f"churn_model-{version}.json", manifest)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump(model, temporary_path)
        os.replace(temporary_path, model_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    _write_json_atomic(registry_dir / "latest.json", manifest)
    return version


def run_pipeline(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    reports_dir: str | Path = REPORTS_DIR,
    registry_dir: str | Path = REGISTRY_DIR,
    *,
    tune: bool = False,
) -> dict[str, Any]:
    cleaned = clean_customer_data(load_data(data_path))
    if cleaned.empty:
        raise ValueError("No usable training rows were found after cleaning.")

    reports_path = Path(reports_dir)
    registry_path = Path(registry_dir)
    champion_path = Path(model_path)
    eda = build_eda_report(cleaned)
    _write_json_atomic(reports_path / "eda.json", eda)

    X, y = _get_features_and_target(cleaned)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if tune:
        from src.tune_model import build_tuned_model

        candidate, training_metrics = build_tuned_model(X_train, y_train)
    else:
        candidate, training_metrics = train_model(cleaned)
    candidate_metrics = _evaluate_on_holdout(candidate, X_test, y_test)

    champion_metrics = None
    if champion_path.is_file():
        champion = joblib.load(champion_path)
        champion_metrics = _evaluate_on_holdout(champion, X_test, y_test)

    promoted = champion_metrics is None or is_better_model(
        candidate_metrics, champion_metrics
    )
    version = None
    if promoted:
        version = _promote_model(
            candidate,
            candidate_metrics,
            champion_path,
            registry_path,
            reason=(
                "Initial model"
                if champion_metrics is None
                else "Improved all required metrics"
            ),
        )

    result: dict[str, Any] = {
        "status": "promoted" if promoted else "rejected",
        "version": version,
        "rows": len(cleaned),
        "tuned": tune,
        "candidate_metrics": candidate_metrics,
        "training_metrics": training_metrics,
        "champion_metrics": champion_metrics,
        "promotion_requires_strict_improvement_in": list(METRICS),
    }
    _write_json_atomic(reports_path / "evaluation.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and register a churn model."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Labeled training CSV.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune the candidate before evaluating it.",
    )
    args = parser.parse_args()
    run_pipeline(data_path=args.data, tune=args.tune)


if __name__ == "__main__":
    main()
