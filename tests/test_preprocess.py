from pathlib import Path

from arabic_nlp_insights.data import read_json_records
from arabic_nlp_insights.preprocess import normalize_arabic


def test_arabic_normalization_removes_diacritics_and_tatweel():
    assert normalize_arabic("إِنَّ العَــرَبِيَّة") == "ان العربية"


def test_normalization_masks_urls_and_mentions():
    text = normalize_arabic("مرحبا @abdalla شاهد https://example.com")
    assert "USER" in text
    assert "URL" in text
    assert "https://" not in text


def test_jsonl_reader(tmp_path: Path):
    path = tmp_path / "rows.json"
    path.write_text(
        '{"input":"أ","output":"Positive"}\n{"input":"ب","output":"Negative"}\n',
        encoding="utf-8",
    )
    rows = read_json_records(path)
    assert len(rows) == 2
    assert rows[0]["output"] == "Positive"


def test_json_array_reader(tmp_path: Path):
    path = tmp_path / "rows.json"
    path.write_text('[{"input":"أ","output":"Positive"}]', encoding="utf-8")
    rows = read_json_records(path)
    assert rows[0]["input"] == "أ"
