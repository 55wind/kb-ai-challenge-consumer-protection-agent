import json

import pytest

from app.corpus.loader import load_corpus
from app.corpus.schema import CaseRecord, CorpusValidationError


def _case_kwargs(**overrides):
    base = dict(
        id="c1",
        title="[TEST-FIXTURE] 사례",
        facts="사실관계",
        issues="쟁점",
        reasoning="판단",
        decision="인용",
        decision_date="2023-01-01",
        category="불완전판매",
        source="[TEST-FIXTURE] pytest",
        source_url="about:test",
    )
    base.update(overrides)
    return base


def test_case_requires_source():
    with pytest.raises(CorpusValidationError):
        CaseRecord.validated(**_case_kwargs(source="", source_url=""))


def test_case_valid():
    r = CaseRecord.validated(**_case_kwargs(decision="기각"))
    assert r.decision == "기각"


def test_case_rejects_unknown_decision():
    with pytest.raises(Exception):
        CaseRecord.validated(**_case_kwargs(decision="보류"))


def test_load_corpus_roundtrip(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_case_kwargs()) + "\n", encoding="utf-8")
    records = load_corpus("cases", corpus_dir=tmp_path)
    assert len(records) == 1
    assert records[0].id == "c1"


def test_load_corpus_fails_on_missing_source(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_case_kwargs(source="")) + "\n", encoding="utf-8")
    with pytest.raises(CorpusValidationError):
        load_corpus("cases", corpus_dir=tmp_path)
