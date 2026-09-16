# Model Card — Arabic NLP Insights

## Systems compared
### Classical baselines
Two logistic-regression pipelines are compared:
- word-level TF-IDF with unigram/bigram features,
- character-level TF-IDF with 3–5 character n-grams.

The development split selects the stronger baseline by macro-F1. That model is then refit on train + development data and evaluated once on the held-out test split.

### Transformer extension
The optional transformer pipeline fine-tunes `aubmindlab/bert-base-arabertv02` for the same source labels. It uses development macro-F1 for checkpoint selection and evaluates the best checkpoint on the held-out test split.

## Metrics
- accuracy,
- macro-F1,
- per-class precision/recall/F1,
- confusion matrix,
- saved misclassification examples for manual error analysis.

## Why compare classical and transformer approaches?
The comparison asks whether the additional complexity and compute of a pretrained Arabic transformer produces meaningful gains over an interpretable sparse-text baseline.

## Current status
No numerical result is claimed until an actual experiment has produced and verified the corresponding artifact.

## Limitations
Sentiment classification can fail on sarcasm, mixed sentiment, dialectal expressions, spelling variation, code-switching, domain-specific vocabulary, and text requiring broader context. Model confidence is not a measure of a person's emotional state.
