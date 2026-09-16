import pandas as pd

from arabic_nlp_insights.baseline import classification_metrics


def test_classification_metrics_reports_macro_f1():
    truth = pd.Series(["Positive", "Negative", "Neutral", "Positive"])
    prediction = ["Positive", "Negative", "Neutral", "Negative"]
    metrics = classification_metrics(truth, prediction)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert set(metrics["labels"]) == {"Positive", "Negative", "Neutral"}
