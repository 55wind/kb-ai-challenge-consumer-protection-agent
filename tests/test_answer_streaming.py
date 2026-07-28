"""답변 스트리밍 — 대기 체감은 줄이되 근거 보장은 그대로여야 한다.

스트리밍은 검증 전에 글자를 화면에 뿌린다. 그래서 "인용 없으면 답하지 않는다"가
깨지기 쉬운 지점이다. 조각을 보냈더라도 확정은 검증 뒤여야 하고, 실패하면
화면에서 지워야 한다.
"""
import pytest

from app.agent import answerer
from app.agent.answerer import Answer, UngroundedAnswer
from app.search.retriever import Hit


class _Rec:
    id = "law-19"
    article = "제19조(설명의무)"
    text = "금융상품판매업자는 중요한 사항을 설명하여야 한다"
    source = "국가법령정보센터"
    source_url = "https://law.go.kr"


def _hits():
    return [Hit(record_id="law-19", score=1.0, record=_Rec())]


class StreamLLM:
    """토큰을 흘리는 LLM. tokens를 통째로 이어붙인 것이 최종 답변."""

    model = "stub"

    def __init__(self, *rounds: str):
        self.rounds = list(rounds)
        self.calls = 0

    def stream(self, system, prompt, timeout=300):
        text = self.rounds[min(self.calls, len(self.rounds) - 1)]
        self.calls += 1
        for ch in text:
            yield ch

    def generate(self, system, prompt, json_mode=False, timeout=120):
        self.calls += 1
        return self.rounds[min(self.calls - 1, len(self.rounds) - 1)]


def test_streams_tokens_then_confirms(monkeypatch):
    """조각이 먼저 흐르고, 마지막에 검증된 Answer가 나온다."""
    monkeypatch.setattr(answerer.settings, "ANSWER_STREAMING", True)
    events = list(answerer.answer_events("설명의무가 뭔가요", _hits(),
                                         StreamLLM("설명의무는 이렇습니다[1]")))
    deltas = [v for k, v in events if k == "delta"]
    finals = [v for k, v in events if k == "answer"]

    assert deltas, "조각이 하나도 흐르지 않았다 — 대기 체감이 그대로다"
    assert "".join(deltas) == "설명의무는 이렇습니다[1]"
    assert len(finals) == 1 and isinstance(finals[0], Answer)
    assert finals[0].citations[0].record_id == "law-19"


def test_ungrounded_stream_is_discarded_and_retried(monkeypatch):
    """인용 없는 생성은 확정하지 않는다 — 지우고 다시 시킨다."""
    monkeypatch.setattr(answerer.settings, "ANSWER_STREAMING", True)
    llm = StreamLLM("근거 없이 쓴 답", "이번엔 근거를 답니다[1]")
    events = list(answerer.answer_events("질문", _hits(), llm))

    kinds = [k for k, _ in events]
    assert "discard" in kinds, "검증 못 한 잠정 표시를 버리라는 신호가 없다"
    assert kinds.index("discard") < kinds.index("answer"), "확정 뒤에 버리면 늦다"
    assert llm.calls == 2, "재시도가 일어나지 않았다"
    final = [v for k, v in events if k == "answer"][0]
    assert final.text == "이번엔 근거를 답니다[1]"


def test_retry_failure_raises(monkeypatch):
    """두 번 다 인용이 없으면 답변으로 내보내지 않는다."""
    monkeypatch.setattr(answerer.settings, "ANSWER_STREAMING", True)
    with pytest.raises(UngroundedAnswer):
        list(answerer.answer_events("질문", _hits(), StreamLLM("근거 없음", "여전히 없음")))


def test_non_streaming_mode_still_works(monkeypatch):
    """킬 스위치를 내리면 이전 동작 — 조각 없이 완성본만."""
    monkeypatch.setattr(answerer.settings, "ANSWER_STREAMING", False)
    events = list(answerer.answer_events("질문", _hits(), StreamLLM("답입니다[1]")))
    assert [k for k, _ in events] == ["answer"]


def test_falls_back_when_llm_cannot_stream(monkeypatch):
    """스트리밍을 지원하지 않는 LLM이어도 답변은 나와야 한다.

    스트리밍은 체감을 줄이는 장치이지 답변의 조건이 아니다.
    """
    monkeypatch.setattr(answerer.settings, "ANSWER_STREAMING", True)

    class NoStream:
        model = "stub"

        def generate(self, system, prompt, json_mode=False, timeout=120):
            return "스트리밍 없이도 답합니다[1]"

    events = list(answerer.answer_events("질문", _hits(), NoStream()))
    assert [k for k, _ in events] == ["answer"]
    assert events[0][1].text == "스트리밍 없이도 답합니다[1]"


def test_orchestrator_emits_delta_and_discard(monkeypatch):
    """오케스트레이터가 조각·폐기를 SSE 이벤트로 중계한다."""
    from app.agent import orchestrator
    from app.agent.classifier import Classification
    from app.agent.orchestrator import SessionState, handle_message

    monkeypatch.setattr(orchestrator.classifier, "classify",
                        lambda t, llm: Classification("예금·계좌", False, "high"))
    monkeypatch.setattr(orchestrator, "retrieve", lambda kind, q, **kw: _hits())
    monkeypatch.setattr(orchestrator, "_expand_by_citation", lambda t, h: h)
    monkeypatch.setattr(orchestrator.answerer, "answer_events",
                        lambda q, h, llm: iter([
                            ("delta", "조각"), ("discard", None),
                            ("delta", "다시"), ("answer", Answer(text="다시[1]", citations=[])),
                        ]))
    types = [e.type for e in handle_message(SessionState(), "질문", StreamLLM("x"))]
    assert types.count("answer_delta") == 2
    assert "answer_discard" in types
    assert "answer" in types
