from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_cleaning import DATA_PATH, clean_customer_data, load_data
from src.feature_engineering import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.pipeline import REPORTS_DIR, _write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "data" / "monitoring" / "predictions.jsonl"
ERRORS_PATH = ROOT / "data" / "monitoring" / "errors.jsonl"
DRIFT_THRESHOLD = 0.2


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as event_file:
        return [json.loads(line) for line in event_file if line.strip()]


def create_reference_profile(data: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for column in NUMERIC_COLUMNS:
        profile["numeric"][column] = {
            "mean": float(data[column].mean()),
            "std": float(data[column].std(ddof=0)),
        }
    for column in CATEGORICAL_COLUMNS:
        values = data[column].fillna("<missing>").astype(str)
        profile["categorical"][column] = values.value_counts(
            normalize=True
        ).to_dict()
    return profile


def calculate_drift(
    reference: dict[str, Any], recent: pd.DataFrame
) -> dict[str, Any]:
    feature_drift: dict[str, dict[str, float]] = {}
    for column, baseline in reference["numeric"].items():
        reference_std = float(baseline["std"])
        scale = reference_std if reference_std > 0 else 1.0
        mean_shift = abs(
            float(recent[column].mean()) - float(baseline["mean"])
        )
        feature_drift[column] = {
            "standardized_mean_shift": mean_shift / scale
        }

    for column, baseline in reference["categorical"].items():
        values = recent[column].fillna("<missing>").astype(str)
        current = values.value_counts(normalize=True)
        categories = set(baseline) | set(current.index)
        feature_drift[column] = {
            "total_variation_distance": 0.5
            * sum(
                abs(
                    float(baseline.get(category, 0.0))
                    - float(current.get(category, 0.0))
                )
                for category in categories
            )
        }

    drifted_features = [
        column
        for column, measures in feature_drift.items()
        if max(measures.values(), default=0.0) >= DRIFT_THRESHOLD
    ]
    return {
        "threshold": DRIFT_THRESHOLD,
        "feature_drift": feature_drift,
        "drift_detected": bool(drifted_features),
        "drifted_features": drifted_features,
    }


def run_monitoring(
    data_path: str | Path = DATA_PATH,
    predictions_path: str | Path = PREDICTIONS_PATH,
    errors_path: str | Path = ERRORS_PATH,
    reports_dir: str | Path = REPORTS_DIR,
    *,
    window: int = 500,
) -> dict[str, Any]:
    if window < 1:
        raise ValueError("Monitoring window must be a positive integer.")
    reference_path = Path(reports_dir) / "reference_profile.json"
    if not reference_path.is_file():
        reference_data = clean_customer_data(load_data(data_path))
        reference_profile = create_reference_profile(reference_data)
        _write_json_atomic(reference_path, reference_profile)

    with reference_path.open(encoding="utf-8") as reference_file:
        reference = json.load(reference_file)
    predictions = _read_events(Path(predictions_path))[-window:]
    recent_errors = _read_events(Path(errors_path))[-window:]
    recent = pd.DataFrame(predictions)
    if not recent.empty:
        feature_columns = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
        recent = recent.reindex(columns=feature_columns)

    report: dict[str, Any] = {
        "checked_at": datetime.now(UTC).isoformat(),
        "window_size": window,
        "prediction_count": len(predictions),
        "error_count": len(recent_errors),
        "drift_detected": False,
        "drifted_features": [],
        "feature_drift": {},
        "status": "insufficient_data" if len(predictions) < 20 else "healthy",
    }
    if len(predictions) >= 20:
        drift = calculate_drift(reference, recent)
        report.update(drift)
        if drift["drift_detected"]:
            report["status"] = "drift_detected"
    if recent_errors:
        report["status"] = "errors_detected"

    _write_json_atomic(Path(reports_dir) / "monitoring.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect prediction logs for feature drift and errors."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Labeled reference training CSV.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="Maximum recent predictions to compare.",
    )
    args = parser.parse_args()
    run_monitoring(data_path=args.data, window=args.window)


if __name__ == "__main__":
    main()
