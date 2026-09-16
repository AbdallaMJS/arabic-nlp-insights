from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REPO = "QCRI/LlamaLens-Arabic"
DATASET_SUBDIR = "ar_reviews_100k"
VALID_SPLITS = {"train", "dev", "test"}


def download_split(split: str, cache_dir: str | Path | None = None) -> Path:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{DATASET_SUBDIR}/{split}.json",
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)


def read_json_records(path: str | Path) -> list[dict]:
    """Read either JSONL or a JSON array without assuming one representation."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Expected a JSON array.")
        return payload

    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
    return rows


def load_split(split: str, cache_dir: str | Path | None = None) -> pd.DataFrame:
    path = download_split(split, cache_dir)
    rows = read_json_records(path)
    frame = pd.DataFrame(rows)
    required = {"input", "output"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required fields: {', '.join(missing)}")
    frame = frame[["input", "output"]].dropna().copy()
    frame["input"] = frame["input"].astype(str)
    frame["output"] = frame["output"].astype(str).str.strip()
    return frame
