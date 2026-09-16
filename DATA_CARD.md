# Data Card — Arabic Reviews 100K via LlamaLens-Arabic

## Source
This project uses the `ar_reviews_100k` sentiment task distributed inside **QCRI/LlamaLens-Arabic** on Hugging Face. The repository does not redistribute the review text; the training code downloads the upstream train/dev/test files when needed.

The LlamaLens-Arabic dataset card reports three sentiment classes with approximately 70k training examples, 20k development examples, and 10k test examples for this task.

License reported by the upstream repository: **CC BY-NC-SA 4.0**. Users of this project remain responsible for following the upstream dataset terms and citation requirements.

## Fields used
Only two task fields are required by this project:
- `input` — Arabic review text
- `output` — sentiment label

Other instruction-format fields distributed by LlamaLens are not needed for the classical baseline.

## Intended use
Educational comparison of:
1. transparent TF-IDF + logistic-regression baselines, and
2. optional AraBERT fine-tuning.

## Evaluation policy
The upstream train/dev/test split is preserved. Development data is used for model selection. The test split is evaluated only after the model choice is made. Macro-F1 is emphasized so each sentiment class contributes equally to the headline metric.

## Limitations and bias
- Reviews cover particular domains and may not represent everyday Arabic conversation.
- Arabic dialects, spelling styles, sarcasm, code-switching, and named entities can affect performance.
- Sentiment labels simplify nuanced opinions into a small number of categories.
- Dataset provenance and annotation quality limit what conclusions can be drawn.
- Performance on this benchmark does not imply equal performance across countries, dialects, or social groups.

## Privacy and responsible use
The project does not attempt to infer protected traits, mental state, or identity. It classifies text into the sentiment categories defined by the source dataset. Error analysis should avoid exposing or redistributing unnecessary personal information from source examples.
