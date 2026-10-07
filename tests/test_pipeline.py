from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.monitoring import calculate_drift
from src.pipeline import is_better_model, run_pipeline


class PromotionGateTests(unittest.TestCase):
    def test_promotion_requires_all_three_metrics_to_improve(self) -> None:
        champion = {"accuracy": 0.8, "f1": 0.7, "roc_auc": 0.9}

        self.assertTrue(
            is_better_model(
                {"accuracy": 0.81, "f1": 0.71, "roc_auc": 0.91}, champion
            )
        )
        self.assertFalse(
            is_better_model(
                {"accuracy": 0.81, "f1": 0.69, "roc_auc": 0.91}, champion
            )
        )
        self.assertFalse(
            is_better_model(
                {"accuracy": 0.8, "f1": 0.71, "roc_auc": 0.91}, champion
            )
        )

    def test_pipeline_registers_initial_model_then_rejects_duplicate(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data_path = project_root / "data" / "customer_churn.csv"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            model_path = temporary_root / "models" / "churn_model.pkl"
            reports_dir = temporary_root / "reports"
            registry_dir = temporary_root / "models" / "registry"
            first_result = run_pipeline(
                data_path,
                model_path,
                reports_dir,
                registry_dir,
            )
            second_result = run_pipeline(
                data_path,
                model_path,
                reports_dir,
                registry_dir,
            )

            self.assertEqual(first_result["status"], "promoted")
            self.assertIsNotNone(first_result["version"])
            self.assertEqual(second_result["status"], "rejected")
            registry_models = list(
                (temporary_root / "models" / "registry").glob("*.pkl")
            )
            self.assertEqual(
                len(registry_models), 1
            )


class DriftDetectionTests(unittest.TestCase):
    def test_feature_distribution_shift_is_detected(self) -> None:
        reference = {
            "numeric": {
                "tenure": {"mean": 12.0, "std": 4.0},
                "monthly_charges": {"mean": 60.0, "std": 10.0},
                "total_charges": {"mean": 720.0, "std": 100.0},
            },
            "categorical": {
                "contract_type": {"Month-to-month": 1.0},
                "internet_service": {"DSL": 1.0},
                "support": {"No": 1.0},
            },
        }
        recent = pd.DataFrame(
            {
                "tenure": [60] * 25,
                "monthly_charges": [60.0] * 25,
                "total_charges": [720.0] * 25,
                "contract_type": ["Two year"] * 25,
                "internet_service": ["DSL"] * 25,
                "support": ["No"] * 25,
            }
        )

        result = calculate_drift(reference, recent)

        self.assertTrue(result["drift_detected"])
        self.assertIn("tenure", result["drifted_features"])
        self.assertIn("contract_type", result["drifted_features"])
