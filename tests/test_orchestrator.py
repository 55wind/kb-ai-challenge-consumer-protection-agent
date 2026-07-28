import pytest

from app.agent import orchestrator
from app.agent.answerer import Answer, Citation
from app.agent.case_outlook import CaseSummary, Outlook
from app.agent.classifier import Classification
from app.agent.redflag import FlagItem, ScanReport
from app.agent.slots import SLOT_SCHEMA, Slots
from app.agent.orchestrator import AgentEvent, SessionState, handle_message
from app.search.retriever import NoEvidenceFound


class StubLLM:
    def generate(self, *a, **k):
        return ""


def _answer():
    return Answer(
        text="답변[1]",
        citations=[Citation("law-19", "제19조", "src", "url", "quote")],
    )


def _events(state, text, monkeypatch_env, mode="chat"):
    return list(handle_message(state, text, StubLLM(), mode=mode))


def test_general_route(monkeypatch):
    monkeypatch.setattr(orchestrator.classifier, "classify",
                        lambda t, llm: Classification("예금·계좌", False, "high"))
    monkeypatch.setattr(orchestrator, "retrieve", lambda kind, q, **kw: ["hit"])
    monkeypatch.setattr(orchestrator.answerer, "answer_events",
                        lambda q, h, llm: iter([("answer", _answer())]))
    events = list(handle_message(SessionState(), "수수료 문의", StubLLM()))
    types = [e.type for e in events]
    assert "classification" in types and "answer" in types and "citations" in types


def test_dispute_route_asks_missing_slot(monkeypatch):
    monkeypatch.setattr(orchestrator.classifier, "classify",
                        lambda t, llm: Classification("불완전판매", True, "high"))
    monkeypatch.setattr(orchestrator.slots_mod, "extract_slots",
                        lambda d, llm: Slots(filled={"who": "본인"},
                                             missing=[k for k in SLOT_SCHEMA if k != "who"]))
    events = list(handle_message(SessionState(), "펀드 사기당했어요", StubLLM()))
    types = [e.type for e in events]
    assert "slots" in types
    assert events[-1].type == "answer"  # 추가 질문


def test_dispute_route_full_slots_gives_outlook(monkeypatch):
    monkeypatch.setattr(orchestrator.classifier, "classify",
                        lambda t, llm: Classification("불완전판매", True, "high"))
    full = {k: "값" for k in SLOT_SCHEMA}
    monkeypatch.setattr(orchestrator.slots_mod, "extract_slots",
                        lambda d, llm: Slots(filled=full, missing=[]))
    monkeypatch.setattr(orchestrator.case_outlook, "outlook",
                        lambda s, llm: Outlook(
                            similar=[CaseSummary("c1", "t", "인용", "r", "s", "u")],
                            stats={"인용": 1, "일부인용": 0, "기각": 0},
                            commentary="유사 사례 1건 중 인용 1건",
                        ))
    events = list(handle_message(SessionState(), "전부 설명했어요", StubLLM()))
    types = [e.type for e in events]
    assert "outlook" in types and "draft_offer" in types


def test_scan_route(monkeypatch):
    monkeypatch.setattr(orchestrator, "detect_input_kind", lambda t: "document")
    monkeypatch.setattr(orchestrator.redflag, "scan",
                        lambda t, llm: ScanReport(items=[FlagItem("원금 보장", "red", "광고", None, "주의")]))
    events = list(handle_message(SessionState(), "원금 보장", StubLLM(), mode="scan"))
    assert events[-1].type == "scan_report"
    assert events[-1].payload["red_count"] == 1
    assert events[-1].payload["input_kind"] == "document"


def test_scan_routes_sms_batch(monkeypatch):
    from app.agent.redflag import ScanSummary

    monkeypatch.setattr(orchestrator, "detect_input_kind", lambda t: "sms_batch")
    monkeypatch.setattr(
        orchestrator.redflag, "scan_sms_batch",
        lambda t, llm: ScanReport(
            items=[FlagItem("사칭 문자", "red", "보이스피싱", None, "신고")],
            summary=ScanSummary(total=1, red=1, yellow=0, top_risk="사칭 문자", actions=["118"]),
            input_kind="sms_batch",
        ),
    )
    events = list(handle_message(SessionState(), "문자들", StubLLM(), mode="scan"))
    assert events[-1].payload["input_kind"] == "sms_batch"
    assert events[-1].payload["summary"]["red"] == 1


def test_scan_routes_transcript(monkeypatch):
    from app.agent.redflag import ScanSummary
    from app.agent.transcript import EvidenceTimeline

    monkeypatch.setattr(orchestrator, "detect_input_kind", lambda t: "transcript")
    monkeypatch.setattr(
        orchestrator.transcript_mod, "mine_evidence",
        lambda t, llm: EvidenceTimeline(items=[], summary=ScanSummary(0, 0, 0, "")),
    )
    state = SessionState()
    events = list(handle_message(state, "직원: ...", StubLLM(), mode="scan"))
    assert events[-1].type == "transcript_report"
    assert state.evidence_timeline is not None


def test_dispute_flow_emits_deadlines_before_outlook(monkeypatch):
    from app.agent.deadlines import DeadlineItem

    monkeypatch.setattr(orchestrator.classifier, "classify",
                        lambda t, llm: Classification("보이스피싱·전자금융사기", True, "high"))
    full = {k: "값" for k in SLOT_SCHEMA}
    monkeypatch.setattr(orchestrator.slots_mod, "extract_slots",
                        lambda d, llm: Slots(filled=full, missing=[]))
    monkeypatch.setattr(
        orchestrator.deadlines_mod, "compute_deadlines",
        lambda cat, when: [DeadlineItem(action="지급정지", period="즉시", computed_deadline=None,
                                        urgency="urgent", law_citation=None, urgency_note="지금")],
    )
    monkeypatch.setattr(orchestrator.case_outlook, "outlook",
                        lambda s, llm: Outlook(similar=[], stats={"인용": 0, "일부인용": 0, "기각": 0},
                                               commentary="유사 사례 없음"))
    events = list(handle_message(SessionState(), "보이스피싱 당했어요", StubLLM()))
    types = [e.type for e in events]
    assert "deadlines" in types
    assert types.index("deadlines") < types.index("outlook")
    assert events[types.index("deadlines")].payload["items"][0]["urgency"] == "urgent"


def test_no_evidence_event(monkeypatch):
    monkeypatch.setattr(orchestrator.classifier, "classify",
                        lambda t, llm: Classification("예금·계좌", False, "high"))

    def raise_no_evidence(*a, **k):
        raise NoEvidenceFound("없음")

    monkeypatch.setattr(orchestrator, "retrieve", raise_no_evidence)
    events = list(handle_message(SessionState(), "무관한 질문", StubLLM()))
    assert events[-1].type == "no_evidence"
    assert "1332" in events[-1].payload["message"]


def test_llm_unavailable_event(monkeypatch):
    from app.llm.ollama_client import LLMUnavailable

    def raise_unavailable(t, llm):
        raise LLMUnavailable("Ollama 연결 불가")

    monkeypatch.setattr(orchestrator.classifier, "classify", raise_unavailable)
    events = list(handle_message(SessionState(), "질문", StubLLM()))
    assert events[-1].type == "error"
