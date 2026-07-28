"""녹취록 증거 채굴: 발화 파싱(결정론) → 직원 발화만 룰 필터 → sLLM 판정
→ 실제 결정례 연결 → EvidenceTimeline (spec §5).
"""
import re
from dataclasses import dataclass, field
from typing import Literal

from app.agent.answerer import Citation
from app.agent.case_outlook import CaseSummary
from app.agent.redflag import _NO_LAW_ADVICE, FlagItem, ScanSummary, _judge, build_summary
from app.agent.rules.sales_rules import match_rules
from app.llm.ollama_client import OllamaClient
from app.search.retriever import NoEvidenceFound, retrieve

# 화자 규칙은 app/agent/speakers.py가 단일 출처다 (판별과 파싱이 공유).
# 여기서는 관대한 파싱 패턴을 쓴다 — 이미 녹취 경로로 확정된 뒤이므로
# 실명·직책 등 어떤 라벨도 받아 조용히 버려지는 줄이 없게 한다.
from app.agent.speakers import NOT_SPEAKER as _NOT_SPEAKER  # noqa: E402
from app.agent.speakers import LINE as _LINE  # noqa: E402
from app.agent.speakers import classify_speaker  # noqa: E402,F401


@dataclass
class Utterance:
    seq: int
    timestamp: str
    speaker: str
    text: str
    is_staff: bool
    role: str = "unknown"


@dataclass
class EvidenceItem:
    seq: int
    timestamp: str
    speaker: str
    utterance: str
    level: Literal["red", "yellow", "check"]
    rule: str
    reason: str
    law_citation: Citation | None
    related_cases: list[CaseSummary] = field(default_factory=list)


@dataclass
class EvidenceTimeline:
    items: list[EvidenceItem] = field(default_factory=list)
    summary: ScanSummary | None = None
    parsed: int = 0            # 발화로 인식한 줄 수
    unparsed: int = 0          # 대화처럼 보이지만 인식하지 못한 줄 수
    scope: str = "staff"       # 'staff' = 직원 발언만 검사, 'all' = 역할 미확정으로 전체 검사
    notice: str = ""           # 사용자에게 보여야 할 해석상 한계


_ROLE_UNKNOWN_NOTICE = (
    "화자 라벨(예: A/B, 실명)만으로는 누가 판매 직원인지 확정할 수 없어 "
    "모든 발언을 검사했습니다. 직원 발언만 보려면 라벨을 '직원'·'고객'으로 바꿔 다시 넣어주세요."
)
_MIXED_ROLE_NOTICE = (
    "일부 화자(예: 실명 라벨)의 역할을 확정하지 못해 그 발언도 함께 검사했습니다. "
    "고객 본인의 발언으로 확인된 것만 검사에서 제외했습니다."
)
_CUSTOMER_ONLY_NOTICE = (
    "고객 발언만 인식되어 검사할 판매자 발언이 없습니다. "
    "검사를 수행하지 못했습니다 — 위험이 없다는 뜻이 아닙니다."
)
_PARSE_FAILED_NOTICE = (
    "녹취록 형식을 인식하지 못했습니다. '직원: 발언' 처럼 화자와 발언을 "
    "콜론(:)으로 구분해 주세요. 검사를 수행하지 못했습니다 — 위험이 없다는 뜻이 아닙니다."
)


# 발화가 아닌 구조적 줄 — 구분선, 순수 타임스탬프, [녹취 시작] 같은 표시
_STRUCTURAL = re.compile(
    r"^\s*(?:"
    r"[-=~_*]{2,}"                              # ────
    r"|[-=~_*]{2,}.*?[-=~_*]{2,}"               # === 녹취 시작 ===
    r"|[\[(]?\d{1,2}:\d{2}(?::\d{2})?[\])]?"    # [00:10]
    r"|[\[(][^\])]*[\])]"                       # (잡음)
    r"|[※#>·•]\s*.*"                            # ※ 메모
    r")\s*$"
)
# '무언가:' 로 시작하는 줄 — 화자 라벨일 가능성이 있다.
# 파싱에 실패했더라도 앞 발화에 이어 붙이면 남의 말을 앞 화자 것으로 만들어버린다.
# 증거로 쓰이는 데이터이므로, 이런 줄은 합치지 않고 '읽지 못함'으로 보고한다.
_LABELISH = re.compile(r"^\s*[^\s:：]{1,20}\s*[:：]\s*\S")


def parse_transcript(text: str) -> tuple[list[Utterance], int]:
    """발화 목록과 '읽지 못한 줄 수'를 함께 돌려준다.

    줄바꿈으로 이어진 발언(앞 줄에 화자, 다음 줄은 본문만)은 앞 발화에 이어 붙인다.
    이걸 버리면 "직원: 이 상품은 원금 보장이고 / 무조건 수익 납니다"에서 위반 문구가
    통째로 사라지면서 미파싱 집계에도 안 잡힌다(콜론이 없으므로).
    """
    utterances: list[Utterance] = []
    unread = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _LINE.match(line)
        speaker = m.group("speaker").strip() if m else ""
        if m and speaker.lower() not in _NOT_SPEAKER:
            role = classify_speaker(speaker)
            utterances.append(Utterance(
                seq=len(utterances) + 1,
                timestamp=m.group("ts") or "",
                speaker=speaker,
                text=m.group("text").strip(),
                is_staff=role == "staff",
                role=role,
            ))
            continue
        if _STRUCTURAL.match(line):
            continue                     # 구분선·타임스탬프 등 — 발화가 아니므로 셈에서 뺀다
        if utterances and not _LABELISH.match(line):
            utterances[-1].text += " " + line.strip()   # 줄바꿈으로 이어진 발언
            continue
        unread += 1                      # 화자를 확정 못 한 줄 — 합치지 않고 보고한다
    return utterances, unread


def count_unparsed(text: str) -> int:
    """읽지 못한 줄 수. parse_transcript가 함께 돌려주는 값과 동일하다.

    콜론 유무로 세던 이전 방식은 두 방향으로 틀렸다 — '※ 녹취 시작 09:30' 같은
    표시를 미파싱으로 부풀리고, 콜론 없는 이어진 발언은 세지 못했다.
    """
    return parse_transcript(text)[1]


def _law_text_and_citation(rule) -> tuple[str, Citation | None]:
    try:
        hits = retrieve("laws", rule.law_query, k=1)
    except NoEvidenceFound:
        return "", None
    r = hits[0].record
    return r.text, Citation(
        record_id=r.id, title=r.article, source=r.source,
        source_url=r.source_url, quote=r.text[:200],
    )


def _related_cases(query: str) -> list[CaseSummary]:
    try:
        hits = retrieve("cases", query, k=2)
    except NoEvidenceFound:
        return []
    return [
        CaseSummary(
            id=h.record_id, title=h.record.title, decision=h.record.decision,
            key_reasoning=h.record.reasoning[:200], source=h.record.source,
            source_url=h.record.source_url,
        )
        for h in hits
    ]


def mine_evidence(text: str, llm: OllamaClient) -> EvidenceTimeline:
    utterances, unparsed = parse_transcript(text)

    if not utterances:
        # 파싱 0건과 "위험 0건"은 전혀 다른 결과다. 구분해서 알린다.
        return EvidenceTimeline(
            items=[], summary=build_summary([], "transcript", unparsed=unparsed),
            parsed=0, unparsed=unparsed, scope="none", notice=_PARSE_FAILED_NOTICE,
        )

    # 검사에서 빼는 것은 '고객 본인의 발언으로 확정된 것'뿐이다.
    # 직원 라벨이 하나라도 있다고 나머지를 버리면, 실명을 쓴 판매 직원의 발언
    # ("김철수: 무조건 원금 보장됩니다")이 아무 표시 없이 사라진다.
    # 확정 못 한 화자는 판매 직원일 수 있으므로 검사하고, 그 사실을 알린다.
    targets = [u for u in utterances if u.role != "customer"]
    unknown_scanned = [u for u in targets if u.role != "staff"]
    if not targets:
        scope, notice = "customer_only", _CUSTOMER_ONLY_NOTICE
    elif not unknown_scanned:
        scope, notice = "staff", ""
    elif any(u.role == "staff" for u in targets):
        scope, notice = "mixed", _MIXED_ROLE_NOTICE
    else:
        scope, notice = "all", _ROLE_UNKNOWN_NOTICE

    items: list[EvidenceItem] = []
    flag_items: list[FlagItem] = []
    for utt in targets:
        rules = match_rules(utt.text)
        if not rules:
            flag_items.append(FlagItem(utt.text, "green", "", None, ""))
            continue
        rule = rules[0]
        rule_label = " · ".join(r.name for r in rules)
        law_text, citation = _law_text_and_citation(rule)
        if citation is None:
            # 근거 조문 없이 위반 판정을 내보내지 않는다 (H-3).
            level, reason = "check", _NO_LAW_ADVICE
            related = []
        else:
            level, reason = _judge(utt.text, rule_label, law_text, llm)
            related = _related_cases(f"{utt.text} {rule.name}")
        flag_items.append(FlagItem(utt.text, level, rule_label, citation, reason))
        items.append(EvidenceItem(
            seq=utt.seq, timestamp=utt.timestamp, speaker=utt.speaker,
            utterance=utt.text, level=level, rule=rule_label, reason=reason,
            law_citation=citation, related_cases=related,
        ))
    return EvidenceTimeline(
        items=items,
        summary=build_summary(flag_items, "transcript", unparsed=unparsed),
        parsed=len(utterances), unparsed=unparsed, scope=scope, notice=notice,
    )
