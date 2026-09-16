# 🗣️ Arabic NLP Insights

Arabic sentiment classification comparing transparent sparse-text baselines with an optional **AraBERT** fine-tuning pipeline.

The project is designed to demonstrate NLP methodology rather than merely call a pretrained model: source-data provenance, Arabic normalization, train/dev/test discipline, baseline comparison, macro-F1 evaluation, per-class analysis, confusion matrices, saved error cases, transformer fine-tuning, and responsible discussion of dialect/domain limitations.

> **Status:** experiment pipelines are ready; numerical results are intentionally not claimed until the corresponding runs have been completed and verified.

## Research question
How much additional value does an Arabic pretrained transformer provide over strong TF-IDF + logistic-regression baselines on a three-class Arabic review-sentiment benchmark?

## Dataset
The project uses the `ar_reviews_100k` task distributed by **QCRI/LlamaLens-Arabic**. The upstream repository provides train, development, and test splits and reports a CC BY-NC-SA 4.0 license. This repository downloads the data on demand and does not redistribute review text.

See `DATA_CARD.md` for provenance, fields, evaluation policy, privacy considerations, and limitations.

## Systems compared
### 1. Word TF-IDF + logistic regression
A transparent word unigram/bigram baseline with class weighting.

### 2. Character TF-IDF + logistic regression
Arabic spelling and morphology make character n-grams a useful comparison. The pipeline uses 3–5 character n-grams.

### 3. AraBERT extension
An optional pipeline fine-tunes `aubmindlab/bert-base-arabertv02` using development macro-F1 for checkpoint selection.

## Arabic preprocessing
The classical baseline applies deliberately light normalization:
- remove Arabic diacritics and tatweel,
- normalize common alef variants,
- normalize alef maqsura / hamza-carrier variants,
- replace URLs and user mentions with placeholders,
- collapse repeated whitespace.

The preprocessing is documented and testable rather than hidden in a notebook.

## Evaluation
The repository emphasizes:
- accuracy,
- **macro-F1**,
- per-class precision/recall/F1,
- confusion matrix,
- saved misclassified examples for qualitative error analysis.

The development set is used for model choice; the held-out test set is used only after that decision.

## Project structure
```text
src/arabic_nlp_insights/
  data.py          upstream dataset download + parsing
  preprocess.py    Arabic normalization
  baseline.py      word/character TF-IDF comparisons
  transformer.py   optional AraBERT fine-tuning
scripts/
  train_baseline.py
  train_transformer.py
app.py             transparent baseline demo
tests/             normalization, parsing, and metric tests
DATA_CARD.md
MODEL_CARD.md
.github/workflows/ci.yml
```

## Quick start — classical baseline
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,app]"
python scripts/train_baseline.py
streamlit run app.py
```

## Optional AraBERT experiment
A GPU/Colab environment is recommended:
```bash
pip install -e ".[transformer]"
python scripts/train_transformer.py --epochs 2
```

For a short pipeline check before full training:
```bash
python scripts/train_transformer.py --epochs 1 --train-limit 3000
```

A limited run is a debugging exercise and must not be presented as the final benchmark result.

## Error-analysis questions
After the genuine experiments run, inspect mistakes by category:
- dialectal vocabulary,
- negation,
- mixed/neutral sentiment,
- sarcasm or irony,
- spelling variation,
- code-switching,
- domain-specific terms,
- unusually long or context-dependent reviews.

The final README should include examples only where doing so is consistent with the source dataset's terms and privacy considerations.

## Responsible interpretation
Sentiment labels are properties of this benchmark, not psychological assessments of people. Performance can differ across dialects, countries, domains, and writing styles. Confidence scores can be wrong and should not be used to infer a person's emotional condition or make consequential decisions.

## Next milestone
Run the classical baselines, verify CI, add genuine held-out metrics and error patterns, then run AraBERT and compare the improvement against additional compute and complexity.
