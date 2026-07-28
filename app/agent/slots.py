"""분쟁성 민원 사실관계 슬롯필링. 추출은 sLLM, 추가 질문은 결정론 템플릿."""
import json
from dataclasses import dataclass, field

from app.llm.ollama_client import OllamaClient

SLOT_SCHEMA = ["who", "when", "product", "incident", "demand"]

_SLOT_DESC = {
    "who": "신청인이 누구인지 (본인/가족/대리인, 일반 소비자인지 사업자인지)",
    "when": "사건이 발생한 시기",
    "product": "관련 금융상품 또는 서비스 (예: 펀드, 대출, 예금, 카드)",
    "incident": "무슨 일이 있었는지 (피해 내용)",
    "demand": "금융회사에 요구하는 것 (예: 배상, 계약 취소, 재발 방지)",
}

# 화면·서류에 쓰는 사람이 읽는 이름 (키는 내부용 영문)
SLOT_LABEL = {
    "who": "누가",
    "when": "언제",
    "product": "상품",
    "incident": "무슨 일",
    "demand": "요구사항",
}

_NEXT_QUESTION = {
    "who": "누가 겪으신 일인가요? 본인이신가요, 가족이나 지인을 대신해 문의하시나요?",
    "when": "언제 있었던 일인가요? 대략적인 시기(연도·월)라도 알려주세요.",
    "product": "어떤 금융상품(예: 펀드, 대출, 예금, 카드, 보험)과 관련된 일인가요?",
    "incident": "어떤 일이 있었는지 조금 더 자세히 설명해 주시겠어요?",
    "demand": "금융회사에 어떤 조치를 원하시나요? (예: 손해배상, 계약 취소, 사과와 재발 방지)",
}


@dataclass
class Slots:
    filled: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    next_question: str | None = None


def extract_slots(dialog: list[dict], llm: OllamaClient) -> Slots:
    """dialog: [{"role": "user"|"assistant", "content": str}, ...]"""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in dialog)
    system = (
        "너는 금융 민원 사실관계 정리 도우미다. 대화에서 아래 항목을 추출해 "
        "반드시 JSON으로만 답한다. 알 수 없는 항목은 빈 문자열로 둔다. 추측 금지.\n"
        + "\n".join(f'- {k}: {v}' for k, v in _SLOT_DESC.items())
        + '\n형식: {"who": "...", "when": "...", "product": "...", "incident": "...", "demand": "..."}'
    )
    raw = llm.generate(system, f"대화:\n{convo}", json_mode=True, timeout=90)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    filled = {
        k: str(data.get(k, "")).strip()
        for k in SLOT_SCHEMA
        if str(data.get(k, "")).strip()
    }
    missing = [k for k in SLOT_SCHEMA if k not in filled]
    return Slots(
        filled=filled,
        missing=missing,
        next_question=_NEXT_QUESTION[missing[0]] if missing else None,
    )
