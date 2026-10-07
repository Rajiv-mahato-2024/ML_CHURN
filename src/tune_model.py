from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.data_cleaning import clean_customer_data
from src.feature_engineering import build_feature_pipeline
from src.train import MODEL_PATH


def build_tuned_model(X_train, y_train):
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
    return search.best_estimator_, {
        "best_cv_f1": float(search.best_score_),
        "best_params": search.best_params_,
    }


def tune_and_save_model(data, path: str | Path = MODEL_PATH):
    cleaned = clean_customer_data(data)
    features = cleaned.drop(columns=["customer_id", "churn"], errors="ignore")
    target = cleaned["churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model, metrics = build_tuned_model(X_train, y_train)
    predictions = model.predict(X_test)
    metrics.update(
        {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(
                y_test, model.predict_proba(X_test)[:, 1]
            ),
        }
    )

    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, metrics


if __name__ == "__main__":
    from src.pipeline import run_pipeline

    run_pipeline(tune=True)
