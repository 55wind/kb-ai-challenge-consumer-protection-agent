"""불공정 레드플래그 스캐너.

조항 단위 분리 → 룰셋 1차 필터 → 룰 히트 조항만 sLLM이 관련 조문과
대조 판정. 룰 미히트 조항은 green (LLM 호출 없음 — CPU 절약).
"""
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from app.agent.answerer import Citation
from app.agent.rules.sales_rules import match_rules
from app.llm.ollama_client import OllamaClient
from app.search.retriever import NoEvidenceFound, retrieve


# level 의미:
#   red/yellow — 근거 조문과 대조해 내린 판정
#   check      — 룰에는 걸렸으나 근거 조문을 찾지 못해 판정을 보류한 상태.
#                "근거 없으면 답하지 않는다"는 원칙을 검진 경로에도 적용한다.
#   green      — 룰 미히트
@dataclass
class FlagItem:
    clause: str
    level: Literal["red", "yellow", "check", "green"]
    rule: str
    law_citation: Citation | None
    advice: str


@dataclass
class ScanSummary:
    total: int
    red: int
    yellow: int
    top_risk: str
    actions: list[str] = field(default_factory=list)
    check: int = 0
    unparsed: int = 0


@dataclass
class ScanReport:
    items: list[FlagItem] = field(default_factory=list)
    summary: "ScanSummary | None" = None
    input_kind: str = "document"

    @property
    def red_count(self) -> int:
        return sum(1 for i in self.items if i.level == "red")


_SMS_REPORT_ACTION = "스팸·스미싱 신고 118, 보이스피싱 112/1332"


def build_summary(items: list[FlagItem], input_kind: str, unparsed: int = 0) -> ScanSummary:
    reds = [i for i in items if i.level == "red"]
    yellows = [i for i in items if i.level == "yellow"]
    checks = [i for i in items if i.level == "check"]
    top = reds[0] if reds else (yellows[0] if yellows else None)
    actions: list[str] = []
    for i in reds + yellows:
        hint = i.advice.split(".")[0].strip()
        if hint and hint not in actions:
            actions.append(hint)
        if len(actions) >= 3:
            break
    if input_kind == "sms_batch" and (reds or yellows):
        actions = actions[:2] + [_SMS_REPORT_ACTION]
    return ScanSummary(
        total=len(items),
        red=len(reds),
        yellow=len(yellows),
        check=len(checks),
        unparsed=unparsed,
        top_risk=top.clause[:40] if top else "",
        actions=actions,
    )


def split_messages(text: str) -> list[str]:
    """문자함 입력: 빈 줄 기준 메시지 블록 분리."""
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


_NO_LAW_ADVICE = (
    "이 문구는 점검 대상 유형에 걸렸지만, 대조할 근거 조문을 찾지 못해 "
    "위반 여부를 판정하지 않았습니다. 금융감독원(1332) 또는 담당자 확인이 필요합니다."
)


def _law_evidence(rule) -> tuple[str, Citation | None]:
    try:
        hits = retrieve("laws", rule.law_query, k=1)
    except NoEvidenceFound:
        return "", None
    r = hits[0].record
    return r.text, Citation(
        record_id=hits[0].record_id, title=r.article, source=r.source,
        source_url=r.source_url, quote=r.text[:200],
    )


def _flag_one(clause: str, llm: OllamaClient) -> FlagItem:
    """조항/문자/발화 1개 판정 (룰 미히트 → green, LLM 미호출)."""
    rules = match_rules(clause)
    if not rules:
        return FlagItem(clause, "green", "", None, "")
    # 한 조항이 복수 유형에 걸리면 전부 보고한다 — 하나만 보면 허위광고+부당권유가
    # 동시에 있는 문구에서 절반이 사라진다 (M-3). 라벨·판정·조언이 같은 집합을 본다.
    rule = rules[0]
    rule_label = " · ".join(r.name for r in rules)
    law_text, citation = _law_evidence(rule)
    if citation is None:
        # 근거 조문 없이 위반 판정을 내보내지 않는다 (H-3).
        return FlagItem(clause, "check", rule_label, None, _NO_LAW_ADVICE)
    level, reason = _judge(clause, rule_label, law_text, llm)
    hints = " ".join(dict.fromkeys(r.hint for r in rules))
    return FlagItem(clause, level, rule_label, citation, f"{hints} {reason}".strip())


def scan_sms_batch(text: str, llm: OllamaClient) -> ScanReport:
    """문자 여러 건 일괄 진단 — 메시지 전체를 조항 1개처럼 취급."""
    items = [_flag_one(msg, llm) for msg in split_messages(text)]
    return ScanReport(
        items=items,
        summary=build_summary(items, "sms_batch"),
        input_kind="sms_batch",
    )


def split_clauses(text: str) -> list[str]:
    """번호·불릿·줄바꿈 기준 조항 분리."""
    parts = re.split(r"\n+|(?=\d+[.)]\s)|(?=[①-⑮])|(?=[-•]\s)", text)
    return [p.strip() for p in parts if p and len(p.strip()) >= 5]


def _judge(clause: str, rule_name: str, law_text: str, llm: OllamaClient) -> tuple[str, str]:
    """rule_name에는 히트한 유형을 전부 넘긴다(' · ' 구분).

    라벨에는 두 유형을 적어놓고 판정은 1순위만 보면, 화면이 실제보다 많은 것을
    검토했다고 말하게 된다. 프롬프트에 전부 넣어 사유가 모든 유형을 다루게 한다.
    """
    system = (
        "너는 금융소비자보호법 위반 심사 도우미다. 반드시 JSON으로만 답한다. "
        '형식: {"level": "red"|"yellow", "reason": "<한두 문장>"}. '
        "red: 조항이 규정 위반 신호에 명확히 해당. yellow: 의심되지만 맥락 확인 필요. "
        "의심 유형이 여러 개면 사유에서 각각을 짚는다."
    )
    prompt = (
        f"검토 대상 문구: {clause}\n\n의심 유형: {rule_name}\n\n"
        f"관련 규정:\n{law_text[:800]}\n\n판정:"
    )
    raw = llm.generate(system, prompt, timeout=90, schema={
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["red", "yellow"]},
            "reason": {"type": "string"},
        },
        "required": ["level", "reason"],
    })
    try:
        data = json.loads(raw)
        level = data.get("level", "yellow")
        if level not in ("red", "yellow"):
            level = "yellow"
        return level, str(data.get("reason", ""))
    except json.JSONDecodeError:
        return "yellow", "자동 판정 실패 — 담당자 확인이 필요합니다."


def scan(document_text: str, llm: OllamaClient) -> ScanReport:
    items = [_flag_one(clause, llm) for clause in split_clauses(document_text)]
    return ScanReport(
        items=items,
        summary=build_summary(items, "document"),
        input_kind="document",
    )
