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
from functools import lru_cache
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
    # 민원 유형 2순위에서 온 카드. "해당될 수 있음"으로 구분해 보여준다 —
    # 1순위 카드와 섞어 놓으면 화면이 확정된 사실처럼 말한다.
    tentative: bool = False


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
    # 청약철회(제46조①)는 **상품 유형별로 기간이 다르다** — 그래서 유형에 붙인다.
    #   1호 보장성 15일 · 2호 투자성 7일 · 3호 대출성 14일
    "불완전판매": [
        DeadlineRule(
            action="투자성 상품 청약철회 (금소법 제46조)",
            period="계약서류 제공일 등부터 7일 이내",
            days=7,
            basis_query="청약의 철회 투자성 상품 7일",
            urgency_note="철회 의사는 서면 등으로 발송한 때 효력이 생깁니다.",
        ),
    ],
    "투자상품": [
        DeadlineRule(
            action="투자성 상품 청약철회 (금소법 제46조)",
            period="계약서류 제공일 등부터 7일 이내",
            days=7,
            basis_query="청약의 철회 투자성 상품 7일",
            urgency_note="철회 의사는 서면 등으로 발송한 때 효력이 생깁니다.",
        ),
    ],
    # 보험(보장성)에 청약철회 카드가 아예 없었다. 코퍼스에서 가장 많은 유형
    # (112건 중 51건)인데도 빠져 있었다. 제46조①1호가 명시한 기간을 넣는다.
    "보험": [
        DeadlineRule(
            action="보장성 상품 청약철회 (금소법 제46조)",
            period="보험증권을 받은 날부터 15일과 청약일부터 30일 중 먼저 오는 날까지",
            days=15,
            basis_query="청약의 철회 보장성 상품 15일 보험증권",
            urgency_note="증권 수령일부터 15일과 청약일부터 30일 중 **먼저 도래하는 날**이 기한입니다.",
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
        # 위법계약해지를 유형에서 떼어 공통으로 올렸다.
        #
        # 제47조①은 적용 대상 위반을 열거한다 — 제17조③(적합성), 제18조②(적정성),
        # 제19조①③(설명의무), 제20조①(불공정영업), 제21조(부당권유). 이 위반들은
        # 보장성·투자성·예금성·대출성 **어느 상품에서도** 일어난다. 그런데 예전에는
        # 이 카드가 `불완전판매`와 `부당권유·불공정영업`에만 달려 있어서, 같은
        # 설명의무 위반이라도 유형이 `보험`이나 `대출·금리`로 분류되면 카드가
        # 사라졌다.
        #
        # 분류 정확도가 완전할 수 없다는 것을 전제하면(현재 금감원 라벨 기준 60%대),
        # **조문이 유형을 가리지 않는 권리는 유형에 묶지 않는 것**이 맞다.
        DeadlineRule(
            action="위법계약 해지 요구 (금소법 제47조)",
            period="위반을 안 날부터 1년, 계약체결일부터 5년 이내",
            days=365,
            basis_query="위법계약의 해지",
            urgency_note="적합성·적정성·설명의무·불공정영업·부당권유 위반이 있었던 경우에 "
                         "해당하며, 위반을 안 시점부터 기간이 진행됩니다.",
        ),
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

# 부당권유·불공정영업은 대출 국면에서도 일어난다 — 제20조①2호(부당한 담보·보증
# 요구), 4호 다목(제3자 연대보증 요구)이 대출성 상품 조항이다. 그래서 투자성만
# 붙이면 대출성 청약철회가 빠진다. 두 유형의 청약철회를 함께 노출한다.
#
# 겹쳐 보여 주는 쪽을 택한 이유: 해당 없는 기한이 하나 더 뜨는 것보다, 해당하는
# 기한이 빠지는 것이 훨씬 나쁘다(09_모델_평가 §3). 각 카드에는 어떤 상품 유형에
# 해당하는지가 제목에 적혀 있어 사용자가 가려 읽을 수 있다.
RULES["부당권유·불공정영업"] = RULES["불완전판매"] + RULES["대출·금리"]

_URGENCY_ORDER = {"urgent": 0, "ok": 1, "unknown": 2, "expired": 3}


def parse_event_date(when_text: str, today: date) -> date | None:
    t = when_text.strip()

    # `2026-07-26` · `2026.7.26` · `2026/07/26`
    #
    # 사용자가 이렇게 쓰는 일도 있지만, 더 흔한 경로는 **슬롯 추출기**다. sLLM은
    # "2026년 7월 26일에 들었는데"를 `{"when": "2026-07-26"}`으로 정규화해서 돌려준다.
    # 이 형태를 못 읽으면, 사용자가 날짜를 분명히 말했는데도 기한이 `unknown`이 되어
    # "시기를 알려주시면 남은 기한을 계산해 드립니다"가 뜬다 — 이 시스템이 막으려는
    # 실패가 시스템 안에서 나는 것이다. 실제로 그렇게 나고 있었다.
    m = re.search(r"(\d{4})\s*[-./]\s*(\d{1,2})(?:\s*[-./]\s*(\d{1,2}))?(?!\s*\d)", t)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))

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


@lru_cache(maxsize=64)
def _law_citation(query: str) -> Citation | None:
    """기한 룰의 근거 조문. **결과를 캐시한다.**

    `basis_query`는 `RULES` 테이블에 박힌 고정 문자열이고 종류가 10여 개뿐인데,
    캐시가 없을 때는 기한 카드마다 하이브리드 검색(임베딩 + FAISS + BM25)을 새로
    돌렸다. 질의 한 건에 카드가 4~6장 뜨므로 사용자 대기 시간에 그만큼이 얹혔다.

    같은 문자열이 같은 조문을 돌려주는 것은 인덱스가 바뀌지 않는 한 보장된다.
    인덱스는 프로세스 기동 때 한 번 적재되므로(`retriever._index_cache`) 캐시의
    수명이 그것과 같다. 관리자 그래프 재구축은 `laws` 인덱스를 바꾸지 않는다.
    """
    try:
        hits = retrieve("laws", query, k=1)
    except NoEvidenceFound:
        return None
    r = hits[0].record
    return Citation(
        record_id=r.id, title=r.article, source=r.source,
        source_url=r.source_url, quote=r.text[:200],
    )


# 상품유형(금소법 제3조) → 청약철회 룰. 제46조①이 유형별로 기간을 달리 정한다.
WITHDRAWAL_BY_PRODUCT: dict[str, DeadlineRule] = {
    "보장성 상품": RULES["보험"][0],
    "투자성 상품": RULES["투자상품"][0],
    "대출성 상품": RULES["대출·금리"][0],
    # 예금성 상품은 제46조① 각 호에 없다 — 청약철회 대상이 아니다.
}


def compute_deadlines(
    category: str, when_text: str, today: date | None = None,
    situation_text: str | None = None, alt_category: str | None = None,
) -> list[DeadlineItem]:
    """법정 기한 카드. 청약철회는 **민원 유형이 아니라 상품유형**으로 고른다.

    제46조①은 상품유형별로 기간을 정한다(보장성 15일·투자성 7일·대출성 14일).
    그런데 예전에는 `classifier.CATEGORIES` 10종으로 골랐다. 그 10종은 상품 축과
    위반 축을 섞어 놓아 배타적이지 않고, 금감원 라벨 기준 정확도가 62.5%였다.
    유형을 잘못 집으면 해당하는 청약철회 카드가 사라진다.

    상품유형 축 하나만 보면 정답 포함률이 96.0%다(`product_type.infer`, 상위 2개).
    그래서 `situation_text`가 주어지면 그것으로 상품유형을 추정해 청약철회 카드를
    붙인다. 없으면 예전처럼 민원 유형으로 고른다 — 기존 호출부와 테스트를 깨지
    않기 위해서다.
    """
    today = today or date.today()
    rules = RULES.get(category, []) + RULES["_common"]

    # 민원 유형 2순위의 기한도 함께 노출한다.
    #
    # 분류 recall@1은 금감원 라벨 기준 57%대인데 recall@2는 72%대다. 즉 정답이
    # 후보에는 있는데 1순위가 아닌 경우가 15%p쯤 된다. 1순위만 쓰면 그 15%p에서
    # 카드가 사라지고, 카드가 사라지는 것은 **보수적 방향이 아니다** — 소비자가
    # 법정 기한을 놓친다. 반대로 하나 더 붙는 것은 사용자가 가려 읽을 수 있다.
    #
    # 그래서 2순위 카드는 `tentative`로 표시해 함께 준다. 확정으로 보이게 섞지
    # 않는 것이 조건이다.
    tentative_actions: set[str] = set()
    if alt_category and alt_category != category:
        known = {r.action for r in rules}
        for r in RULES.get(alt_category, []):
            if r.action not in known:
                rules = rules + [r]
                known.add(r.action)
                tentative_actions.add(r.action)

    if situation_text:
        from app.agent import product_type

        try:
            kinds = product_type.infer(situation_text)
        except Exception as e:  # noqa: BLE001 - 임베딩 실패가 기한 안내를 막아선 안 된다
            print(f"[product_type] 상품유형 추정 실패 — 민원 유형으로 대체합니다: "
                  f"{type(e).__name__}: {e}", flush=True)
        else:
            # 청약철회는 상품유형이 정한다. 유형 기반으로 붙어 있던 것은 걷어낸다.
            rules = [r for r in rules if "청약철회" not in r.action]
            seen = {r.action for r in rules}
            for k in kinds:
                rule = WITHDRAWAL_BY_PRODUCT.get(k)
                if rule and rule.action not in seen:
                    rules.append(rule)
                    seen.add(rule.action)
                    # 상품유형이 확정한 카드다. 2순위 유형에도 같은 카드가 있었다는
                    # 이유로 「해당될 수 있음」을 달면, 조문이 정한 기간을 추측처럼
                    # 보여주게 된다 — 근거의 무게가 다르다.
                    tentative_actions.discard(rule.action)
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
        tentative = rule.action in tentative_actions
        if tentative:
            note = "이 유형에 해당될 경우의 기한입니다. " + note
        items.append(DeadlineItem(
            action=rule.action, period=rule.period, computed_deadline=deadline,
            urgency=urgency, law_citation=citation, urgency_note=note,
            tentative=tentative,
        ))
    items.sort(key=lambda i: _URGENCY_ORDER[i.urgency])
    return items
