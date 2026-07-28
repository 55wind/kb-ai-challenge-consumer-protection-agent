"""골든타임 카운트다운: 유형별 법정 기한 룰 + 결정론 날짜 계산.

원칙(spec §3): 날짜는 정규식으로만 파싱(LLM 금지). 룰 일수는 laws 코퍼스
조문 원문과 대조 검증된 값만 수록 — 코퍼스로 검증 불가한 기한(예: 통신사기
피해환급법상 서면신청 기한)은 days=None으로 두고 행동 안내만 한다.
모든 항목에 기산점 주의 문구를 동봉한다 (법률 단정 회피).

"영업일" 기한(business_days=True)은 주말을 제외해 가산한다. 공휴일 테이블은
외부 데이터 의존이라 범위 밖 — 그래서 계산 결과는 실제 기한보다 이르거나 같고,
사용자에게 그 사실을 문구로 함께 고지한다.
"""
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from app.agent.answerer import Citation
from app.search.retriever import NoEvidenceFound, retrieve

_CAVEAT = " 기산점에 따라 달라질 수 있으니 금융감독원(1332)으로 확인하세요."


@dataclass(frozen=True)
class DeadlineRule:
    action: str
    period: str
    days: int | None  # 0=즉시형, None=계산 불가(안내만)
    basis_query: str
    urgency_note: str
    business_days: bool = False  # True면 역일이 아닌 영업일로 가산


# 영업일 가산은 주말(토·일)만 제외한다. 관공서·금융 공휴일 테이블은 매년 갱신되는
# 외부 데이터 의존이라 범위에 넣지 않았다 — 공휴일이 끼면 실제 기한은 이 계산보다
# 하루 이상 뒤로 밀리므로, 안내는 항상 실제 기한보다 이르거나 같다(보수적).
_BUSINESS_DAY_CAVEAT = " (주말 제외로 계산했습니다. 공휴일이 끼면 실제 기한은 하루 이상 늦춰질 수 있습니다.)"


def add_business_days(start: date, days: int) -> date:
    """start 다음 영업일부터 세어 days번째 영업일을 반환 (초일불산입 관행)."""
    current = start
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # 월(0)~금(4)
            remaining -= 1
    return current


@dataclass
class DeadlineItem:
    action: str
    period: str
    computed_deadline: str | None
    urgency: Literal["expired", "urgent", "ok", "unknown"]
    law_citation: Citation | None
    urgency_note: str


# 일수 검증 근거 (2026-07-23, data/corpus/laws.jsonl 원문 대조):
# - 금소법 제46조 ①2: 투자성 상품 — 계약서류 제공일 등부터 7일
# - 금소법 제46조 ①3: 대출성 상품 — 계약서류 수령일 등부터 14일
# - 금소법 제47조 ①: 위법계약 해지 — 5년 이내의 대통령령 기간(안 날부터 1년은 시행령)
# - 통신사기피해환급법 제3조 ①: 사기이용계좌 지급정지 등 피해구제 신청 (즉시형)
# - 동법 시행령 제3조 ②: 전화·구술 신청 시 신청한 날부터 3영업일 이내 서면 제출
RULES: dict[str, list[DeadlineRule]] = {
    "보이스피싱·전자금융사기": [
        DeadlineRule(
            action="해당 은행 콜센터·경찰(112)에 사기이용계좌 지급정지 신청 (통신사기피해환급법 제3조)",
            period="즉시 (골든타임)",
            days=0,
            basis_query="전기통신금융사기 사기이용계좌 지급정지 피해구제의 신청",
            urgency_note="지금 바로 전화하세요. 1분이 피해액을 좌우합니다.",
        ),
        DeadlineRule(
            action="피해구제신청서와 신분증 사본을 은행에 서면 제출 (전화로 신청한 경우)",
            period="전화·구술 신청일부터 3영업일 이내",
            days=3,
            basis_query="피해구제 신청 서면 신청서류 3영업일 이내 제출",
            urgency_note="기간 내 서면 제출이 없으면 신청이 없었던 것으로 봅니다.",
            business_days=True,
        ),
    ],
    "불완전판매": [
        DeadlineRule(
            action="위법계약 해지 요구 (금소법 제47조)",
            period="위반을 안 날부터 1년, 계약체결일부터 5년 이내",
            days=365,
            basis_query="위법계약의 해지",
            urgency_note="설명의무 위반 등을 안 시점부터 기간이 진행됩니다.",
        ),
        DeadlineRule(
            action="투자성 상품 청약철회 (금소법 제46조)",
            period="계약서류 제공일 등부터 7일 이내",
            days=7,
            basis_query="청약의 철회 투자성 상품 7일",
            urgency_note="철회 의사는 서면 등으로 발송한 때 효력이 생깁니다.",
        ),
    ],
    "대출·금리": [
        DeadlineRule(
            action="대출성 상품 청약철회 (금소법 제46조)",
            period="계약서류 수령일 등부터 14일 이내",
            days=14,
            basis_query="청약의 철회 대출성 상품 14일",
            urgency_note="철회 시 원리금 반환으로 계약이 소멸합니다.",
        ),
    ],
    "_common": [
        DeadlineRule(
            action="[1단계] 은행 소비자보호 민원 접수 (KB국민은행 1588-9999·영업점)",
            period="지체 없이 — 은행 자체 민원 처리가 첫 단계",
            days=None,
            basis_query="은행 민원 접수 처리 프로세스",
            urgency_note="은행 단계에서 해결되면 가장 빠릅니다. 불수용 시 금융감독원 분쟁조정으로 넘어가세요.",
        ),
        DeadlineRule(
            action="[2단계] 금융감독원 분쟁조정 신청",
            period="기한 제한은 없으나 지연될수록 증거 확보가 어려움",
            days=None,
            basis_query="분쟁조정 신청",
            urgency_note="관련 자료(계약서·녹취·문자)를 지금부터 모아두세요.",
        ),
    ],
}

# "부당권유·불공정영업"도 불완전판매 룰 공유 (위법계약해지 대상 위반 유형)
RULES["부당권유·불공정영업"] = RULES["불완전판매"]

_URGENCY_ORDER = {"urgent": 0, "ok": 1, "unknown": 2, "expired": 3}


def parse_event_date(when_text: str, today: date) -> date | None:
    t = when_text.strip()

    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월(?:\s*(\d{1,2})\s*일)?", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
        return _safe_date(y, mo, d)

    m = re.search(r"(작년|올해|재작년)\s*(\d{1,2})\s*월(?:\s*(\d{1,2})\s*일)?", t)
    if m:
        y = today.year - {"올해": 0, "작년": 1, "재작년": 2}[m.group(1)]
        return _safe_date(y, int(m.group(2)), int(m.group(3) or 1))

    m = re.search(r"(\d{1,2})\s*[/월]\s*(\d{1,2})\s*일?", t)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        candidate = _safe_date(today.year, mo, d)
        if candidate and candidate > today:
            candidate = _safe_date(today.year - 1, mo, d)
        return candidate

    m = re.search(r"(\d+)\s*개월\s*전", t)
    if m:
        months = int(m.group(1))
        y, mo = today.year, today.month - months
        while mo <= 0:
            mo += 12
            y -= 1
        return _safe_date(y, mo, min(today.day, 28))

    m = re.search(r"(\d+)\s*일\s*전", t)
    if m:
        return today - timedelta(days=int(m.group(1)))

    if "지난달" in t or "지난 달" in t:
        y, mo = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        return _safe_date(y, mo, min(today.day, 28))
    if "오늘" in t:
        return today
    if "어제" in t:
        return today - timedelta(days=1)
    return None


def _safe_date(y: int, mo: int, d: int) -> date | None:
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _law_citation(query: str) -> Citation | None:
    try:
        hits = retrieve("laws", query, k=1)
    except NoEvidenceFound:
        return None
    r = hits[0].record
    return Citation(
        record_id=r.id, title=r.article, source=r.source,
        source_url=r.source_url, quote=r.text[:200],
    )


def compute_deadlines(
    category: str, when_text: str, today: date | None = None
) -> list[DeadlineItem]:
    today = today or date.today()
    rules = RULES.get(category, []) + RULES["_common"]
    event = parse_event_date(when_text or "", today)
    items: list[DeadlineItem] = []
    for rule in rules:
        citation = _law_citation(rule.basis_query)
        if rule.days == 0:
            urgency, deadline = "urgent", None
        elif rule.days is None or event is None:
            urgency, deadline = "unknown", None
        else:
            dl = (add_business_days(event, rule.days) if rule.business_days
                  else event + timedelta(days=rule.days))
            deadline = dl.isoformat()
            remaining = (dl - today).days
            urgency = "expired" if remaining < 0 else ("urgent" if remaining <= 7 else "ok")
        note = rule.urgency_note + _CAVEAT
        # 영업일 산정 한계 문구는 실제로 계산했을 때만 붙인다.
        # 기산점을 몰라 계산을 못 했는데 "주말 제외로 계산했습니다"라고 하면
        # 하지 않은 계산을 했다고 말하는 셈이다.
        if rule.business_days and deadline is not None:
            note += _BUSINESS_DAY_CAVEAT
        if urgency == "unknown" and rule.days is not None:
            note = "시기를 알려주시면 남은 기한을 계산해 드립니다. " + note
        items.append(DeadlineItem(
            action=rule.action, period=rule.period, computed_deadline=deadline,
            urgency=urgency, law_citation=citation, urgency_note=note,
        ))
    items.sort(key=lambda i: _URGENCY_ORDER[i.urgency])
    return items
