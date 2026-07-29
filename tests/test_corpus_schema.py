import json

import pytest

from app.corpus.loader import load_corpus
from app.corpus.schema import CaseRecord, CorpusValidationError, InterpretationDoc


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


def _interp_kwargs(**over):
    base = dict(
        id="i1", source="[TEST-FIXTURE] pytest", source_url="about:test",
        doc_type="FAQ", topic="Ⅰ. 시험", title="시험 문항인가?",
        text="□ 그렇다.", cited_articles=["제19조"],
    )
    return {**base, **over}


def test_interpretation_requires_source():
    with pytest.raises(CorpusValidationError):
        InterpretationDoc.validated(**_interp_kwargs(source="", source_url=""))


def test_interpretation_rejects_unknown_doc_type():
    """FAQ와 해설만 허용한다 — 화면이 이 값으로 출처 표기를 달리하기 때문이다.

    조문(법)과 유권해석(당국 견해)은 효력이 다르다. 새 값이 조용히 들어오면
    구분이 무너진 채로 인용 화면에 나간다.
    """
    with pytest.raises(Exception):
        InterpretationDoc.validated(**_interp_kwargs(doc_type="유권해석"))


def test_real_interpretations_corpus_loads():
    """실제 코퍼스가 스키마를 통과하는지. 파싱 스크립트 회귀를 여기서 잡는다."""
    records = load_corpus("interpretations")
    assert len(records) > 100, "유권해석 코퍼스가 비었거나 파싱이 깨졌다"
    assert {r.doc_type for r in records} == {"FAQ", "해설"}
    assert all(r.title.strip() and r.text.strip() for r in records)
