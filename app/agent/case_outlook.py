"""유사 분쟁조정 사례 전망.

원칙(설계서 §7.2): 확률 수치를 만들지 않는다. 실제 결정례 top-k의
인용/일부인용/기각 분포와 사례별 논거를 사실 그대로 서술한다.
stats 집계는 코드가 하고, sLLM은 서술문만 작성한다.
"""
import re
from dataclasses import dataclass, field

from app.agent.slots import Slots
from app.llm.ollama_client import OllamaClient
from app.search.retriever import Hit, retrieve

_FORBIDDEN = re.compile(r"(확률|%|퍼센트|가능성\s*[0-9]|승소율)")


@dataclass
class CaseSummary:
    id: str
    title: str
    decision: str
    key_reasoning: str
    source: str
    source_url: str


@dataclass
class Outlook:
    similar: list[CaseSummary] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    commentary: str = ""
    caveat: str = ""  # 통계적 한계 고지 — UI가 항상 표시


def _summarize_query(slots: Slots) -> str:
    return " ".join(slots.filled.get(k, "") for k in ("product", "incident", "demand")).strip()


def outlook(slots: Slots, llm: OllamaClient, k: int = 5) -> Outlook:
    query = _summarize_query(slots)
    hits: list[Hit] = retrieve("cases", query, k=k)
    similar = [
        CaseSummary(
            id=h.record_id,
            title=h.record.title,
            decision=h.record.decision,
            key_reasoning=h.record.reasoning[:300],
            source=h.record.source,
            source_url=h.record.source_url,
        )
        for h in hits
    ]
    # 코퍼스 월간 갱신 때 새 결정 라벨(예: "각하", "조정성립")이 들어와도 죽지 않는다.
    # 알 수 없는 라벨은 "기타"로 모아 화면에 그대로 드러낸다.
    # 다만 이건 2차 방어다 — 1차는 schema.py의 decision Literal이고, 새 라벨은
    # 코퍼스 로딩에서 먼저 거부된다. 그 실패는 /api/health의 corpus 체크가 잡는다.
    stats = {"인용": 0, "일부인용": 0, "기각": 0, "기타": 0}
    for s in similar:
        key = s.decision if s.decision in stats else "기타"
        stats[key] += 1

    case_lines = "\n".join(
        f"- {s.title} → {s.decision}. 논거: {s.key_reasoning[:150]}" for s in similar
    )
    system = (
        "너는 금융분쟁 사례 분석 도우미다. 규칙:\n"
        "1. 아래 실제 조정 사례들의 결정 결과와 논거를 소비자 눈높이로 요약한다.\n"
        "2. 절대 확률, 퍼센트, 승소율 같은 수치 예측을 만들지 않는다.\n"
        "3. '유사 사례 N건 중 인용 M건' 같은 사실 서술만 한다.\n"
        "4. 각 사례에서 어떤 사정이 인용/기각을 갈랐는지 짚어준다."
    )
    # 합계가 사례 수와 맞아야 한다 — 기타는 0이면 생략하고 있으면 반드시 포함한다.
    dist = f"인용 {stats['인용']}, 일부인용 {stats['일부인용']}, 기각 {stats['기각']}"
    if stats["기타"]:
        dist += f", 기타 {stats['기타']}"
    prompt = (
        f"소비자 상황: {query}\n\n"
        f"유사 사례 {len(similar)}건 ({dist}):\n"
        f"{case_lines}\n\n전망 서술:"
    )
    commentary = llm.generate(system, prompt, timeout=180).strip()
    if _FORBIDDEN.search(commentary):
        # 금칙 표현 재생성 1회, 재발 시 결정론 문장으로 대체
        commentary = llm.generate(
            system, prompt + "\n(확률·퍼센트 표현 없이 다시 서술)", timeout=180
        ).strip()
        if _FORBIDDEN.search(commentary):
            commentary = (
                f"유사한 실제 조정 사례 {len(similar)}건 중 인용 {stats['인용']}건, "
                f"일부인용 {stats['일부인용']}건, 기각 {stats['기각']}건"
                + (f", 기타 {stats['기타']}건" if stats["기타"] else "")
                + "입니다. 각 사례의 결정 논거를 함께 확인해 보세요."
            )
    from app.search.retriever import _get_records

    corpus_size = len(_get_records("cases"))
    caveat = (
        f"수집된 금융감독원 공개 결정례 {corpus_size}건 중 유사 사례 {len(similar)}건의 "
        "결과일 뿐, 전체 분쟁 통계나 결과 예측이 아닙니다. 실제 결과는 개별 사실관계에 따라 달라집니다."
    )
    return Outlook(similar=similar, stats=stats, commentary=commentary, caveat=caveat)
