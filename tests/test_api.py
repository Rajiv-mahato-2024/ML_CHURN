from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import main


class PredictionApiTests(unittest.TestCase):
    def test_routes_and_telco_form_payload(self) -> None:
        telco_form = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.0,
            "TotalCharges": 840.0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            event_path = Path(temporary_directory) / "predictions.jsonl"
            with patch.object(main, "PREDICTIONS_PATH", event_path):
                home_response = main.home()
                health_response = main.health()
                prediction_response = main.predict(main.Customer(**telco_form))
                event_was_recorded = event_path.is_file()

        self.assertIn("message", home_response)
        self.assertEqual(health_response["status"], "healthy")
        self.assertIn("prediction", prediction_response)
        self.assertIn("churn_probability", prediction_response)
        self.assertTrue(event_was_recorded)

    def test_prediction_accepts_canonical_feature_names(self) -> None:
        response = main.predict(
            main.Customer.model_validate(
                {
                    "tenure": 12,
                    "monthly_charges": 70.0,
                    "total_charges": 840.0,
                    "contract_type": "Month-to-month",
                    "internet_service": "Fiber",
                    "support": "No",
                }
            )
        )

        self.assertIn("prediction", response)
        self.assertIn("churn_probability", response)
