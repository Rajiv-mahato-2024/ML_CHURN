from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data_cleaning import clean_customer_data, load_or_create_dataset
from src.feature_engineering import build_feature_pipeline
from src.train import MODEL_PATH


def tune_and_save_model(data, path: str | Path = MODEL_PATH):
    cleaned = clean_customer_data(data)
    if "customer_id" in cleaned.columns:
        features = cleaned.drop(columns=["customer_id", "churn"])
    else:
        features = cleaned.drop(columns=["churn"])
    target = cleaned["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_feature_pipeline()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    model = search.best_estimator_
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "best_cv_f1": search.best_score_,
        "best_params": search.best_params_,
    }

    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, metrics


if __name__ == "__main__":
    dataset = load_or_create_dataset()
    _, metrics = tune_and_save_model(dataset)
    print(f"Tuned model saved to: {MODEL_PATH}")
    print(f"Best cross-validation F1: {metrics['best_cv_f1']:.3f}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Test F1: {metrics['f1']:.3f}")
    print(f"Best parameters: {metrics['best_params']}")
