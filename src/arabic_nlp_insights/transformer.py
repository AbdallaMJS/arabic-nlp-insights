from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "aubmindlab/bert-base-arabertv02"


def fine_tune_arabert(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path = "artifacts/arabert",
    epochs: int = 2,
    max_length: int = 192,
    train_limit: int | None = None,
):
    """Fine-tune AraBERT and evaluate once on the held-out test split."""
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    labels = sorted(train["output"].astype(str).unique().tolist())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    if train_limit:
        train = train.iloc[:train_limit].copy()

    def to_dataset(frame: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {
                "text": frame["input"].astype(str).tolist(),
                "label": [label2id[str(label)] for label in frame["output"]],
            }
        )

    train_ds = to_dataset(train)
    dev_ds = to_dataset(dev)
    test_ds = to_dataset(test)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    dev_ds = dev_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    def compute_metrics(eval_prediction):
        logits, label_ids = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(label_ids, predictions),
            "macro_f1": f1_score(label_ids, predictions, average="macro"),
        }

    output = Path(output_dir)
    arguments = TrainingArguments(
        output_dir=str(output),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    trainer.save_model(str(output / "best_model"))
    tokenizer.save_pretrained(str(output / "best_model"))
    return {"labels": labels, "test_metrics": test_metrics, "model_name": MODEL_NAME}
