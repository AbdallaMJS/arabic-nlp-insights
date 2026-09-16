from __future__ import annotations

import argparse

from arabic_nlp_insights.data import load_split
from arabic_nlp_insights.transformer import fine_tune_arabert


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AraBERT on Arabic review sentiment")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--output", default="artifacts/arabert")
    args = parser.parse_args()

    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")
    report = fine_tune_arabert(
        train,
        dev,
        test,
        output_dir=args.output,
        epochs=args.epochs,
        train_limit=args.train_limit,
    )
    print(report)


if __name__ == "__main__":
    main()
