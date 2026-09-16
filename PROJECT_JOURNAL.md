# Arabic NLP Insights — Project Journal

> **Authenticity note:** This journal template was prepared with AI assistance. Abdalla should complete each dated entry only after personally running the work described. Do not backdate entries or copy results that were not reproduced locally.

## How to use this journal

For every work session, record the real date, commands run, outputs checked, decisions made, problems encountered, and what was learned. Link the related GitHub issue/comment, evidence file, and commit when available.

---

## Session 1 — Classical Arabic NLP baselines

**Date:**

**Environment / machine:**

**Commands I personally ran:**

```text

```

**What I verified:**
- [ ] Dataset files and label set checked
- [ ] Train / development / test split checked
- [ ] Word TF-IDF + logistic regression baseline trained
- [ ] Character TF-IDF + logistic regression baseline trained
- [ ] Development macro-F1 recorded for both
- [ ] Selected baseline evaluated once on held-out test data

**Selected baseline and reason:**

**Held-out baseline metrics:**

**Evidence links / screenshots:**

**Problems I encountered and how I solved them:**

**What I learned in my own words:**

**Related GitHub issue/comment:**

**Related commit:**

---

## Session 2 — AraBERT fine-tuning and comparison

**Date:**

**GPU / Colab environment:**

**Training configuration I used:**

**Commands / notebook cells I personally ran:**

```text

```

**AraBERT held-out metrics:**

**Classical baseline vs AraBERT comparison:**

**Compute / complexity differences I observed:**

**Evidence links:**

**What I learned in my own words:**

**Related commit:**

---

## Session 3 — Error analysis and responsible interpretation

**Date:**

**Misclassification examples reviewed:**

**Observed error categories (only where supported by examples):**
- [ ] Negation
- [ ] Dialect variation
- [ ] Spelling variation
- [ ] Mixed sentiment
- [ ] Code-switching
- [ ] Sarcasm / irony
- [ ] Other: __________________________

**What the errors suggest:**

**README / model-card updates made:**

**Streamlit demo tested:**
- [ ] Baseline model loads
- [ ] Arabic input works
- [ ] Prediction output is understandable
- [ ] Limitations are visible

**CI status checked:**

**Evidence links:**

**Final commit:**

---

## Reproducibility sign-off

- [ ] I can recreate the environment from the repository instructions.
- [ ] I can explain TF-IDF and logistic regression.
- [ ] I can explain why character n-grams can help with Arabic text variation.
- [ ] I can explain macro-F1 and why development/test separation matters.
- [ ] I can explain what transfer learning with AraBERT changes.
- [ ] I can reproduce the baseline evaluation.
- [ ] I can explain the observed error patterns without inventing causes.
- [ ] I can describe at least two dataset/model limitations without reading a script.

**Abdalla initials:**

**Date:**
