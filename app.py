from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

from arabic_nlp_insights.preprocess import normalize_arabic

MODEL_PATH = Path("artifacts/baseline/model.joblib")

st.set_page_config(page_title="Arabic NLP Insights", page_icon="🗣️", layout="centered")
st.title("🗣️ Arabic NLP Insights")
st.caption("Arabic sentiment classification with transparent baselines and careful error analysis")

st.markdown(
    "This reviewer demo uses the selected TF-IDF + logistic-regression baseline. "
    "A separate training pipeline can fine-tune AraBERT for comparison."
)

if not MODEL_PATH.exists():
    st.info(
        "No verified model artifact is committed. Run `python scripts/train_baseline.py`, "
        "review the held-out metrics, and then launch this app locally."
    )
else:
    model = joblib.load(MODEL_PATH)
    text = st.text_area("Arabic text", height=140, placeholder="اكتب مراجعة باللغة العربية...")
    if st.button("Classify sentiment", type="primary") and text.strip():
        prediction = model.predict([text])[0]
        st.metric("Predicted label", str(prediction))
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            labels = model.classes_
            st.bar_chart({str(label): float(probability) for label, probability in zip(labels, probabilities)})
        with st.expander("Normalized text used by the baseline"):
            st.write(normalize_arabic(text))
        st.caption(
            "Sentiment labels are dataset categories, not a psychological assessment. "
            "Dialect, sarcasm, code-switching, and domain shift can cause errors."
        )
