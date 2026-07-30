import pytest

from app.agent import case_outlook
from app.agent.case_outlook import Outlook, outlook
from app.agent.draft import build_draft
from app.agent.slots import Slots
from app.corpus.schema import CaseRecord
from app.search.retriever import Hit


class StubLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def generate(self, system, prompt, json_mode=False, timeout=120, schema=None):
        return self.responses.pop(0) if self.responses else ""


def _case_hit(i: int, decision: str) -> Hit:
    rec = CaseRecord.validated(
        id=f"fx-{i}", title=f"[TEST-FIXTURE] 사례{i}", facts="사실", issues="쟁점",
        reasoning="설명의무 위반이 인정되어 손해배상 책임이 있다",
        decision=decision, decision_date="2023-01-01", category="불완전판매",
        source="[TEST-FIXTURE] pytest", source_url="about:test",
    )
    return Hit(record_id=rec.id, score=1.0, record=rec)


def _slots() -> Slots:
    return Slots(filled={
        "who": "본인", "when": "2025년 3월", "product": "사모펀드",
        "incident": "위험 설명 없이 가입 권유받아 손실", "demand": "손해배상",
    })


@pytest.fixture
def _stub_retrieve(monkeypatch):
    hits = [_case_hit(1, "인용"), _case_hit(2, "인용"), _case_hit(3, "기각")]
    monkeypatch.setattr(case_outlook, "retrieve", lambda kind, q, k: hits[:k])


def test_outlook_stats_counted_in_code(_stub_retrieve):
    result = outlook(_slots(), StubLLM(["유사 사례 3건 중 인용 2건입니다."]), k=3)
    assert isinstance(result, Outlook)
    assert result.stats == {"인용": 2, "일부인용": 0, "기각": 1, "기타": 0}
    assert len(result.similar) == 3


def test_outlook_survives_unknown_decision_label(_stub_retrieve, monkeypatch):
    """코퍼스 갱신으로 새 결정 라벨이 들어와도 500이 아니라 '기타'로 집계된다 (M-2)."""
    import app.agent.case_outlook as co

    original = co.retrieve

    def patched(kind, query, k=5):
        hits = original(kind, query, k=k)
        hits[0].record.decision = "각하"          # 3종 밖 라벨
        return hits

    monkeypatch.setattr(co, "retrieve", patched)
    result = outlook(_slots(), StubLLM(["사실 서술"]), k=3)
    assert result.stats["기타"] == 1
    assert sum(result.stats.values()) == 3


def test_unknown_label_counted_in_deterministic_fallback(_stub_retrieve, monkeypatch):
    """금칙어로 결정론 문장이 나갈 때도 합계가 사례 수와 맞아야 한다."""
    import app.agent.case_outlook as co

    original = co.retrieve

    def patched(kind, query, k=5):
        hits = original(kind, query, k=k)
        hits[0].record.decision = "각하"
        return hits

    monkeypatch.setattr(co, "retrieve", patched)
    # 두 번 다 확률 표현 → 결정론 문장으로 대체
    result = outlook(_slots(), StubLLM(["인용 확률 90%", "승소율 80%"]), k=3)
    assert "기타 1건" in result.commentary
    assert "확률" not in result.commentary


def test_outlook_filters_probability_language(_stub_retrieve):
    llm = StubLLM(["인용 확률 87%입니다.", "이번에도 승소율 90%!"])
    result = outlook(_slots(), llm, k=3)
    # 2회 모두 금칙어 → 결정론 문장으로 대체
    assert "확률" not in result.commentary
    assert "%" not in result.commentary
    assert "3건" in result.commentary


def test_draft_fills_template():
    draft = build_draft(_slots(), StubLLM(["2025년 3월 위험 설명 없이 가입을 권유받아 손실을 입었다."]),
                        category="불완전판매", respondent="○○은행")
    assert "○○은행" in draft.markdown
    assert "사모펀드" in draft.markdown
    assert "- [ ] " in draft.markdown  # 증빙 체크리스트
    assert "녹취" in draft.markdown  # 불완전판매 증빙
    assert "(작성 필요)" not in draft.markdown.split("첨부")[0].replace("(금융회사명 작성 필요)", "")


def test_draft_appends_evidence_timeline():
    from app.agent.redflag import ScanSummary
    from app.agent.transcript import EvidenceItem, EvidenceTimeline

    ev = EvidenceTimeline(
        items=[EvidenceItem(seq=1, timestamp="[00:10]", speaker="직원",
                            utterance="원금 보장입니다", level="red",
                            rule="허위·과장광고 의심 (금소법 제22조)",
                            reason="단정", law_citation=None, related_cases=[])],
        summary=ScanSummary(total=1, red=1, yellow=0, top_risk="원금 보장입니다"),
    )
    draft = build_draft(_slots(), StubLLM([""]), category="불완전판매",
                        respondent="○○은행", evidence_timeline=ev)
    assert "별첨" in draft.markdown and "원금 보장입니다" in draft.markdown
    assert "타임라인 (별첨, 1건)" in draft.markdown


def test_draft_without_evidence_unchanged():
    draft = build_draft(_slots(), StubLLM([""]), category="불완전판매")
    assert "별첨:" not in draft.markdown


def test_draft_marks_missing_slots():
    slots = Slots(filled={"product": "펀드"}, missing=["who", "when", "incident", "demand"])
    draft = build_draft(slots, StubLLM([]), category="기타")
    assert "(작성 필요)" in draft.markdown
