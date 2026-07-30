import pytest

from app.agent import redflag
from app.agent.redflag import ScanReport, scan, split_clauses
from app.agent.rules.sales_rules import match_rules
from app.corpus.schema import LawChunk
from app.search.retriever import Hit


class StubLLM:
    def __init__(self, response='{"level": "red", "reason": "명백한 위반 신호"}'):
        self.response = response
        self.calls = 0

    def generate(self, system, prompt, json_mode=False, timeout=120, schema=None):
        self.calls += 1
        return self.response


@pytest.fixture(autouse=True)
def _stub_law_retrieve(monkeypatch):
    rec = LawChunk.validated(
        id="law-22", law_name="[TEST-FIXTURE] 금소법", article="제22조(광고규제)",
        text="누구든지 금융상품 광고에서 손실 보전을 단정해서는 안 된다.",
        source="[TEST-FIXTURE] pytest", source_url="about:test",
    )
    monkeypatch.setattr(
        redflag, "retrieve", lambda kind, q, k: [Hit("law-22", 1.0, rec)]
    )


def test_match_rules_flags_guaranteed_principal():
    rules = match_rules("이 투자상품은 원금 보장에 확정 수익 10%를 드립니다")
    assert any("광고" in r.name for r in rules)


def test_match_rules_ignores_benign():
    assert match_rules("영업시간은 오전 9시부터 오후 4시까지입니다") == []


def test_split_clauses():
    parts = split_clauses("1. 원금 보장 상품입니다\n2. 수수료는 없습니다")
    assert len(parts) == 2


def test_scan_marks_red_and_green():
    llm = StubLLM()
    report = scan("1. 원금 100% 보장 확정 수익!\n2. 영업시간은 9시부터입니다", llm)
    assert isinstance(report, ScanReport)
    levels = [i.level for i in report.items]
    assert "red" in levels and "green" in levels
    red = next(i for i in report.items if i.level == "red")
    assert red.law_citation is not None
    assert llm.calls == 1  # green 조항은 LLM 호출 없음


def test_scan_downgrades_on_bad_llm_json():
    report = scan("원금 보장 상품", StubLLM("판정 불가"))
    assert report.items[0].level == "yellow"
