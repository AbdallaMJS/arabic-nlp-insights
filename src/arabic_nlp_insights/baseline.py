from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from .preprocess import normalize_arabic

RANDOM_STATE = 42


def candidate_pipelines() -> dict[str, Pipeline]:
    common_model = dict(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    return {
        "word_tfidf_logreg": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=normalize_arabic,
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=120_000,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", LogisticRegression(**common_model)),
            ]
        ),
        "char_tfidf_logreg": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=normalize_arabic,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=150_000,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", LogisticRegression(**common_model)),
            ]
        ),
    }


def classification_metrics(y_true, y_pred) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "labels": labels,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_baselines(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path = "artifacts/baseline",
) -> dict:
    required = {"input", "output"}
    for name, frame in {"train": train, "dev": dev, "test": test}.items():
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} split must contain input and output columns")

    candidates = candidate_pipelines()
    dev_scores = {}
    for name, pipeline in candidates.items():
        pipeline.fit(train["input"], train["output"])
        predictions = pipeline.predict(dev["input"])
        dev_scores[name] = classification_metrics(dev["output"], predictions)

    best_name = max(candidates, key=lambda name: dev_scores[name]["macro_f1"])
    best_model = candidates[best_name]

    combined = pd.concat([train, dev], ignore_index=True)
    best_model.fit(combined["input"], combined["output"])
    test_predictions = best_model.predict(test["input"])
    test_metrics = classification_metrics(test["output"], test_predictions)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output / "model.joblib")

    errors = test.copy()
    errors["prediction"] = test_predictions
    errors = errors[errors["output"] != errors["prediction"]]
    errors.head(500).to_csv(output / "error_analysis.csv", index=False)

    report = {
        "selected_baseline": best_name,
        "development_metrics": dev_scores,
        "test_metrics": test_metrics,
        "split_sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "note": "Report only metrics produced by a verified run of this pipeline.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
