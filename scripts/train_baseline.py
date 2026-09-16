from __future__ import annotations

from arabic_nlp_insights.baseline import train_baselines
from arabic_nlp_insights.data import load_split


def main() -> None:
    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")
    report = train_baselines(train, dev, test)
    print("Selected baseline:", report["selected_baseline"])
    print("Test metrics:", report["test_metrics"])


if __name__ == "__main__":
    main()
