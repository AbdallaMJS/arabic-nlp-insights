from __future__ import annotations

import re

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION = re.compile(r"@[\w_]+")
WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Light normalization that preserves words while reducing orthographic noise."""
    text = str(text)
    text = URL.sub(" URL ", text)
    text = MENTION.sub(" USER ", text)
    text = text.replace(TATWEEL, "")
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = WHITESPACE.sub(" ", text).strip()
    return text
