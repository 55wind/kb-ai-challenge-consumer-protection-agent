"""민원유형 분류: 임베딩 유사도로 후보 2개 → sLLM이 확정 + 분쟁성 판정."""
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.llm.ollama_client import OllamaClient
from app.search import embedder

CATEGORIES: dict[str, str] = {
    "불완전판매": "펀드, 신탁, 투자상품 등을 가입할 때 위험 설명을 듣지 못했거나 잘못된 설명을 들은 경우, 설명의무 위반",
    "보이스피싱·전자금융사기": "보이스피싱, 스미싱, 명의도용, 계좌 부정이체, 전자금융 사기 피해",
    "대출·금리": "대출 거절, 금리 인하 요구, 이자 계산, 연체, 근저당, 상환 관련 문의와 분쟁",
    "부당권유·불공정영업": "상품 가입 강요, 꺾기, 부당한 권유, 우월적 지위 남용",
    "예금·계좌": "예금, 적금, 계좌 개설과 해지, 수수료 관련 문의",
    "카드": "신용카드, 체크카드, 할부, 청약철회, 카드 부정사용",
    "보험": "보험금 지급, 보험 계약, 상해·질병·자동차 보험 관련 분쟁",
    "투자상품": "주식, 펀드, 파생상품 등 투자 관련 일반 문의",
    "절차문의": "민원 접수 방법, 분쟁조정 신청 절차, 필요 서류, 처리 기간 문의",
    "기타": "위 유형에 해당하지 않는 금융 관련 문의",
}

_cat_vecs: np.ndarray | None = None


@dataclass
class Classification:
    category: str
    is_dispute: bool
    confidence: Literal["high", "low"]


def _candidates(text: str, top: int = 2) -> list[str]:
    global _cat_vecs
    names = list(CATEGORIES)
    if _cat_vecs is None:
        _cat_vecs = embedder.embed([f"{n}: {d}" for n, d in CATEGORIES.items()])
    qv = embedder.embed([text])[0]
    sims = _cat_vecs @ qv
    order = np.argsort(sims)[::-1][:top]
    return [names[i] for i in order]


def classify(text: str, llm: OllamaClient) -> Classification:
    cands = _candidates(text)
    system = (
        "너는 금융 민원 분류기다. 반드시 JSON으로만 답한다. "
        '형식: {"category": "<후보 중 하나>", "is_dispute": true|false}. '
        "is_dispute는 금융회사와의 다툼·피해구제 요구(분쟁성)이면 true, 단순 정보 문의면 false."
    )
    # 후보를 이름만 주지 않고 **설명과 함께** 준다.
    #
    # 이름만 주면(`['불완전판매', '보험']`) 모델이 그 이름의 뜻을 스스로 추측해야
    # 한다. 예컨대 '불완전판매'가 설명의무 위반을 가리킨다는 것은 이 분류 체계의
    # 약속이지 낱말에서 자명하게 나오는 뜻이 아니다. 설명은 CATEGORIES에 이미
    # 있으니 안 줄 이유가 없다.
    #
    # 결정례 60건 실측: 80.0% → 83.3% (`scripts/experiment_classifier.py`).
    # 후보를 고르는 단계는 건드리지 않으므로 회수율에는 영향이 없다.
    listed = "\n".join(f"- {c}: {CATEGORIES[c]}" for c in cands)
    prompt = f"민원 내용: {text}\n\n후보 유형:\n{listed}\n\n어느 유형인가?"
    raw = llm.generate(system, prompt, json_mode=True, timeout=60)
    try:
        data = json.loads(raw)
        category = data.get("category", "")
        if category not in CATEGORIES:
            raise ValueError(category)
        return Classification(
            category=category,
            is_dispute=bool(data.get("is_dispute", False)),
            confidence="high",
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        # sLLM 응답 불량 → 임베딩 1순위 후보로 폴백
        return Classification(category=cands[0], is_dispute=False, confidence="low")
