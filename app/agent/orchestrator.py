"""결정론적 에이전트 오케스트레이터.

흐름 제어는 전부 이 모듈의 코드가 담당하고, sLLM은 각 단계(분류·추출·
생성·판정)만 수행한다. handle_message는 AgentEvent 제너레이터를 반환해
서버가 SSE로 중계한다.
"""
import traceback
from dataclasses import dataclass, field
from typing import Iterator, Literal

from app.agent import (
    answerer,
    case_outlook,
    classifier,
    deadlines as deadlines_mod,
    draft as draft_mod,
    redflag,
    slots as slots_mod,
    transcript as transcript_mod,
)
from app.agent.answerer import UngroundedAnswer
from app.agent.detector import detect_input_kind
from app.config import settings
from app.graph import expand
from app.graph.runtime import curated_graph
from app.llm.ollama_client import LLMUnavailable, OllamaClient
from app.search.retriever import Hit, NoEvidenceFound, retrieve

DISCLAIMER = "본 서비스는 법률 자문이 아닌 정보 제공입니다. 정확한 판단은 금융감독원(1332) 또는 전문가와 상담하세요."

EventType = Literal[
    "status", "answer", "citations", "outlook", "slots", "draft_offer",
    "scan_report", "glossary", "no_evidence", "error", "classification",
    "deadlines", "transcript_report", "busy", "answer_delta", "answer_discard",
]


@dataclass
class AgentEvent:
    type: EventType
    payload: dict = field(default_factory=dict)


# 세션별 대화·슬롯 상태 (서버가 세션당 하나씩 유지)
@dataclass
class SessionState:
    dialog: list[dict] = field(default_factory=list)
    category: str = ""
    slots_filled: dict[str, str] = field(default_factory=dict)
    evidence_timeline: "transcript_mod.EvidenceTimeline | None" = None


def _citations_payload(citations) -> list[dict]:
    return [
        {
            "record_id": c.record_id,
            "title": c.title,
            "source": c.source,
            "source_url": c.source_url,
            "quote": c.quote,
        }
        for c in citations
    ]


def handle_message(
    state: SessionState,
    text: str,
    llm: OllamaClient,
    mode: Literal["chat", "scan"] = "chat",
) -> Iterator[AgentEvent]:
    state.dialog.append({"role": "user", "content": text})
    try:
        if mode == "scan":
            yield from _handle_scan(state, text, llm)
        else:
            yield from _handle_chat(state, text, llm)
    except LLMUnavailable as e:
        yield AgentEvent("error", {"message": str(e)})
    except NoEvidenceFound:
        yield AgentEvent(
            "no_evidence",
            {
                "message": "확인된 근거를 찾지 못했습니다. 추측으로 답하지 않습니다. "
                "금융감독원 콜센터(국번없이 1332) 또는 e-금융민원센터에 문의해 보세요."
            },
        )


def _summary_payload(summary) -> dict | None:
    if summary is None:
        return None
    return {
        "total": summary.total,
        "red": summary.red,
        "yellow": summary.yellow,
        "check": summary.check,
        "unparsed": summary.unparsed,
        "top_risk": summary.top_risk,
        "actions": summary.actions,
    }


def _scan_report_event(report) -> AgentEvent:
    return AgentEvent(
        "scan_report",
        {
            "red_count": report.red_count,
            "input_kind": report.input_kind,
            "summary": _summary_payload(report.summary),
            "items": [
                {
                    "clause": i.clause,
                    "level": i.level,
                    "rule": i.rule,
                    "advice": i.advice,
                    "law_citation": _citations_payload([i.law_citation])[0] if i.law_citation else None,
                }
                for i in report.items
            ],
        },
    )


def _handle_scan(state: SessionState, text: str, llm: OllamaClient) -> Iterator[AgentEvent]:
    kind = detect_input_kind(text)
    if kind == "transcript":
        yield AgentEvent("status", {"message": "녹취록으로 인식했습니다. 발언을 검토하고 있습니다..."})
        timeline = transcript_mod.mine_evidence(text, llm)
        state.evidence_timeline = timeline
        yield AgentEvent(
            "transcript_report",
            {
                "summary": _summary_payload(timeline.summary),
                "parsed": timeline.parsed,
                "unparsed": timeline.unparsed,
                "scope": timeline.scope,
                "notice": timeline.notice,
                "items": [
                    {
                        "seq": i.seq,
                        "timestamp": i.timestamp,
                        "speaker": i.speaker,
                        "utterance": i.utterance,
                        "level": i.level,
                        "rule": i.rule,
                        "reason": i.reason,
                        "law_citation": _citations_payload([i.law_citation])[0] if i.law_citation else None,
                        "related_cases": [
                            {
                                "id": c.id, "title": c.title, "decision": c.decision,
                                "key_reasoning": c.key_reasoning, "source": c.source,
                                "source_url": c.source_url,
                            }
                            for c in i.related_cases
                        ],
                    }
                    for i in timeline.items
                ],
            },
        )
        return
    if kind == "sms_batch":
        yield AgentEvent("status", {"message": "문자함으로 인식했습니다. 문자별로 검사하고 있습니다..."})
        yield _scan_report_event(redflag.scan_sms_batch(text, llm))
        return
    yield AgentEvent("status", {"message": "문서를 조항 단위로 검토하고 있습니다..."})
    yield _scan_report_event(redflag.scan(text, llm))


def _handle_chat(state: SessionState, text: str, llm: OllamaClient) -> Iterator[AgentEvent]:
    yield AgentEvent("status", {"message": "문의 유형을 파악하고 있습니다..."})
    cls = classifier.classify(text, llm)
    state.category = cls.category
    yield AgentEvent("classification", {"category": cls.category, "is_dispute": cls.is_dispute})

    if cls.category == "절차문의":
        yield from _procedure_answer(state, text, llm)
    elif cls.is_dispute:
        yield from _dispute_flow(state, text, llm)
    else:
        yield from _general_answer(state, text, llm)


def _expand_by_citation(text: str, hits: list) -> list:
    """조문 검색이 놓친 근거를, 유사 결정례가 인용한 조문으로 보강한다.

    소비자는 조문의 말로 묻지 않는다. 같은 상황의 결정례는 그 조문을 원문에서
    인용하고 있으므로, 그 인용을 따라가면 검색이 못 찾은 근거에 닿는다.
    연결의 출처는 결정례 원문의 정규식 추출이라 근거로 제시할 수 있다.

    실패해도 원래 근거를 그대로 돌려준다 — 부가 레이어이지 의존 대상이 아니다.
    """
    if not settings.GRAPH_EXPANSION:
        return hits
    try:
        graph = curated_graph()
        if graph is None:
            return hits
        cases = retrieve("cases", text, k=3)
        titles = {h.record_id: getattr(h.record, "title", "") for h in cases}
        found = expand.cited_laws(
            graph,
            [h.record_id for h in cases],
            limit=settings.GRAPH_EXPANSION_MAX,
            exclude={h.record_id for h in hits},
        )
        if not found:
            return hits
        records = _law_records()
        added: list[Hit] = []
        for c in found:
            if c.law_id in records:
                # 코퍼스에 본문이 있는 조문 — 조문 원문 그대로
                added.append(Hit(record_id=c.law_id, score=0.0, record=records[c.law_id]))
                continue
            # 코퍼스 밖 법령(민법·자본시장법 등). 본문은 없지만 위원회가 그 법리를
            # 서술한 문장이 결정례 원문에 있다 — 출처를 결정례로 밝혀 근거로 쓴다.
            rec = expand.as_record(c, titles.get(c.cited_by, ""))
            if rec is not None:
                added.append(Hit(record_id=c.law_id, score=0.0, record=rec))
        if added:
            print(f"[graph] 근거 확장 {len(added)}건: "
                  f"{', '.join(a.record_id for a in added)}", flush=True)
        return hits + added
    except NoEvidenceFound:
        return hits  # 유사 결정례가 없으면 확장할 것도 없다
    except Exception:  # noqa: BLE001 - 확장 실패가 답변을 막아서는 안 된다
        traceback.print_exc()
        return hits


def _law_records() -> dict:
    from app.search.retriever import _get_records

    return _get_records("laws")


def _add_evidence(hits: list[Hit], kind: str, text: str, k: int) -> list[Hit]:
    """보조 근거를 뒤에 덧붙인다. 실패해도 답변을 막지 않는다.

    근거가 없는 것(NoEvidenceFound)과 검색이 고장 난 것을 구분해서 다룬다.
    전자는 정상이라 조용히 넘기고, 후자는 로그를 남긴다 — 조용히 넘기면 그
    근거가 통째로 빠진 답변이 계속 나가는데 운영자는 기능이 죽은 줄 모른다.
    """
    try:
        return hits + retrieve(kind, text, k=k)
    except NoEvidenceFound:
        return hits  # 이 질문과 관련된 문서가 없다 — 정상
    except Exception as e:  # noqa: BLE001
        print(f"[{kind}] 검색 실패 — 이 근거 없이 답변합니다: "
              f"{type(e).__name__}: {e}", flush=True)
        return hits


def _general_answer(state: SessionState, text: str, llm: OllamaClient,
                    kind: str = "laws") -> Iterator[AgentEvent]:
    yield AgentEvent("status", {"message": "관련 규정과 사례를 찾고 있습니다..."})
    hits = retrieve(kind, text, k=4)
    if kind == "laws":
        # 법적 근거를 앞에 모은다 — 조문, 그다음 결정례가 인용한 조문.
        # KB 상품 문서는 질문에 따라 소음이 되므로 뒤로 보낸다. 소형 모델은
        # 앞쪽 근거에 주목하므로 순서가 곧 우선순위다.
        hits = _expand_by_citation(text, hits[:3])

    # 감독당국 해석은 **두 경로 모두에** 붙인다. 조문은 "무엇이 금지되는가"만
    # 말하고, "그래서 내 경우가 여기 해당하는가"는 FAQ·설명자료에 문장으로 적혀
    # 있다.
    #
    # 법령 경로에만 붙였다가 실제로 답을 놓쳤다. "설명서를 종이로 안 주고 앱
    # 화면으로만 보여줬는데 이게 맞나요?"가 `절차문의`로 분류돼 절차 경로로 갔고,
    # 민원 접수 절차 문서 네 건만 근거로 잡혀 "그런 언급은 없습니다"라고 답했다.
    # 정답(10문10답 4번 — 전자적 장치 화면 표시도 설명서 제공에 포함)은 코퍼스에
    # 있었는데 경로가 그것을 못 보게 막고 있었다. 분류가 흔들려도 근거가 사라지지
    # 않게 하려면 이쪽이 맞다.
    hits = _add_evidence(hits, "interpretations", text, k=2)
    if kind == "laws":
        # KB 약관은 당국 해석 뒤에 둔다 — 효력 순서다.
        hits = _add_evidence(hits, "kb_products", text, k=2)
    try:
        ans = None
        for kind, value in answerer.answer_events(text, hits, llm):
            if kind == "delta":
                yield AgentEvent("answer_delta", {"text": value})
            elif kind == "discard":
                # 재시도 전 잠정 표시를 버린다 — 검증 못 한 글을 남기지 않는다
                yield AgentEvent("answer_discard", {})
            else:
                ans = value
        # 조각을 먼저 보냈더라도 확정은 인용 검증을 통과한 지금이다
        assert ans is not None
    except UngroundedAnswer:
        # 재시도까지 실패했다. 잠정 표시를 지우고 근거 없음으로 마무리한다.
        yield AgentEvent("answer_discard", {})
        yield AgentEvent(
            "no_evidence",
            {"message": "근거를 갖춘 답변을 만들지 못했습니다. 금융감독원(1332)에 문의해 보세요."},
        )
        return
    state.dialog.append({"role": "assistant", "content": ans.text})
    yield AgentEvent("answer", {"text": ans.text})
    yield AgentEvent("citations", {"items": _citations_payload(ans.citations)})
    if ans.glossary:
        yield AgentEvent(
            "glossary",
            {"items": [{"term": g.term, "definition": g.definition} for g in ans.glossary]},
        )


def _procedure_answer(state: SessionState, text: str, llm: OllamaClient) -> Iterator[AgentEvent]:
    yield from _general_answer(state, text, llm, kind="procedures")


def _dispute_flow(state: SessionState, text: str, llm: OllamaClient) -> Iterator[AgentEvent]:
    yield AgentEvent("status", {"message": "사실관계를 정리하고 있습니다..."})
    slots = slots_mod.extract_slots(state.dialog, llm)
    # 이전 턴에 채운 슬롯 유지 병합
    slots.filled = {**state.slots_filled, **slots.filled}
    slots.missing = [k for k in slots_mod.SLOT_SCHEMA if k not in slots.filled]
    state.slots_filled = slots.filled
    yield AgentEvent("slots", {"filled": slots.filled, "missing": slots.missing})

    if slots.missing:
        question = slots_mod._NEXT_QUESTION[slots.missing[0]]
        state.dialog.append({"role": "assistant", "content": question})
        yield AgentEvent("answer", {"text": question})
        return

    deadline_items = deadlines_mod.compute_deadlines(
        state.category, slots.filled.get("when", "")
    )
    if deadline_items:
        yield AgentEvent(
            "deadlines",
            {
                "items": [
                    {
                        "action": d.action,
                        "period": d.period,
                        "computed_deadline": d.computed_deadline,
                        "urgency": d.urgency,
                        "urgency_note": d.urgency_note,
                        "law_citation": _citations_payload([d.law_citation])[0] if d.law_citation else None,
                    }
                    for d in deadline_items
                ]
            },
        )

    yield AgentEvent("status", {"message": "유사한 실제 분쟁조정 사례를 찾고 있습니다..."})
    ol = case_outlook.outlook(slots, llm)
    yield AgentEvent(
        "outlook",
        {
            "stats": ol.stats,
            "commentary": ol.commentary,
            "caveat": ol.caveat,
            "similar": [
                {
                    "id": s.id, "title": s.title, "decision": s.decision,
                    "key_reasoning": s.key_reasoning, "source": s.source,
                    "source_url": s.source_url,
                }
                for s in ol.similar
            ],
        },
    )
    state.dialog.append({"role": "assistant", "content": ol.commentary})
    yield AgentEvent(
        "draft_offer",
        {"message": "사실관계가 정리되었습니다. 분쟁조정 신청서 초안을 만들어 드릴까요?"},
    )


def build_draft_for(state: SessionState, llm: OllamaClient, respondent: str = "") -> AgentEvent:
    slots = slots_mod.Slots(
        filled=state.slots_filled,
        missing=[k for k in slots_mod.SLOT_SCHEMA if k not in state.slots_filled],
    )
    d = draft_mod.build_draft(slots, llm, category=state.category, respondent=respondent,
                              evidence_timeline=state.evidence_timeline)
    return AgentEvent("draft_offer", {
        "markdown": d.markdown,
        "evidence": d.evidence_checklist,
        "missing_fields": d.missing_fields,
    })
