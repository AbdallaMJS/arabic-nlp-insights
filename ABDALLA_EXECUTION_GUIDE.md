# Abdalla Execution Guide — Arabic NLP Insights

This file is a **student execution guide**, not evidence that the experiments have already been completed. The branch was prepared with AI-assisted development support. Abdalla should personally run the experiments, inspect the errors, and complete `PROJECT_JOURNAL.md` in his own words.

## Goal
Reproduce the Arabic sentiment-classification pipeline, compare transparent TF-IDF + logistic-regression baselines, then compare the selected classical baseline with AraBERT using verified held-out metrics and qualitative error analysis.

## 1. Work on the execution branch
```bash
git clone https://github.com/AbdallaMJS/arabic-nlp-insights.git
cd arabic-nlp-insights
git checkout abdalla-execution
```

## 2. Create a clean environment and install the classical pipeline
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Then:
```bash
python -m pip install --upgrade pip
pip install -e ".[dev,app]"
```

## 3. Quality checks
```bash
ruff check src tests scripts app.py
pytest -q
```
Save the terminal output.

## 4. Run the classical baselines
```bash
python scripts/train_baseline.py
```
The pipeline compares word TF-IDF and character TF-IDF on the development set, selects by development macro-F1, retrains the selected system on train+dev, and evaluates once on the held-out test set.

Inspect:
- `artifacts/baseline/metrics.json`
- `artifacts/baseline/error_analysis.csv`
- `artifacts/baseline/model.joblib`

Record the real split sizes, both development macro-F1 values, selected baseline, held-out test accuracy/macro-F1, and per-class precision/recall/F1.

## 5. Verify the experimental discipline
Abdalla should confirm that:
- the test split was not used to choose between word and character baselines;
- macro-F1 is used because all classes should matter rather than only the majority class;
- the baseline preprocessing is deliberately light and reproducible.

## 6. Perform genuine error analysis
Review a representative sample from `error_analysis.csv`. Categorize only patterns supported by the actual examples, such as:
- negation;
- dialectal vocabulary;
- spelling variation;
- mixed/neutral sentiment;
- code-switching;
- sarcasm/irony;
- long/context-dependent reviews.

Do not force every error into a category, and do not republish large amounts of source review text.

## 7. Run AraBERT in a GPU/Colab environment
Install the transformer extras:
```bash
pip install -e ".[transformer]"
```
For a pipeline check only:
```bash
python scripts/train_transformer.py --epochs 1 --train-limit 3000
```
Do **not** report the limited run as the final benchmark.

Then run the intended experiment:
```bash
python scripts/train_transformer.py --epochs 2
```
Record the real configuration, runtime/device, validation behavior, and held-out metrics produced by the script.

## 8. Compare classical baseline vs AraBERT
Create a compact comparison covering:
- held-out accuracy;
- held-out macro-F1;
- per-class behavior;
- common error patterns;
- compute/runtime requirements;
- model complexity and reproducibility trade-offs.

If AraBERT does not improve macro-F1, report that honestly.

## 9. Run the reviewer demo
For the classical model:
```bash
streamlit run app.py
```
Test several short Arabic inputs and note that demo predictions are benchmark-model outputs, not psychological assessments.

## 10. Final student-owned evidence
After the real runs, Abdalla should personally add or approve:
- completed `PROJECT_JOURNAL.md`;
- a verified baseline-vs-AraBERT comparison table;
- confusion matrix / class-level summary;
- error-analysis findings based on actual cases;
- README/model-card updates with only verified metrics and limitations.

Suggested final student commit:
```text
Document verified Arabic NLP baseline and AraBERT comparison
```

## Interview check
Abdalla should be able to answer without notes:
1. What is TF-IDF and why can it be a strong baseline?
2. Why can character n-grams help Arabic text classification?
3. Why use macro-F1?
4. Why must model selection happen on development rather than test data?
5. What does AraBERT add compared with sparse features?
6. What kinds of Arabic-language variation can still cause errors?
