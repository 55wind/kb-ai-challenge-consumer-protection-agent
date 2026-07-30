"""민원유형 분류: 임베딩 유사도로 후보 2개 → sLLM이 확정 + 분쟁성 판정."""
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.llm.ollama_client import LLMUnavailable, OllamaClient
from app.search import embedder

# 유형 설명은 **금소법 조문의 열거를 그대로 옮긴 것**을 기본으로 한다.
#
# 처음에는 우리가 감각으로 쓴 문장이었다. 그러다 금융위 「금소법 FAQ(3차)」의
# 「Ⅰ. 불공정영업행위 관련」 문항 5개를 프로브로 넣었더니 하나도 회수하지
# 못했다. 원인은 단순했다 — 그 章의 문항이 전부 **연대보증**에 관한 것인데,
# 우리 `부당권유·불공정영업` 설명에는 연대보증이라는 말이 없었다. 제20조제1항이
# 제2호("부당하게 담보를 요구하거나 보증을 요구하는 행위")와 제4호 다목
# ("제3자의 연대보증을 요구하는 경우")으로 명시하고 있는데도 우리 문장에서
# 빠져 있었던 것이다.
#
# 그래서 우리가 무엇을 빠뜨렸는지 판단하는 대신 **조문이 열거한 것을 옮긴다.**
# 이 설명은 임베딩으로 후보를 뽑는 데 그대로 쓰이므로, 조문에 있는 말이
# 들어 있어야 그 말로 묻는 질의가 회수된다.
CATEGORIES: dict[str, str] = {
    # 제19조(설명의무) · 제17조(적합성원칙) · 제18조(적정성원칙)
    "불완전판매": "펀드, 신탁, 투자상품 등을 가입할 때 상품 내용·투자 위험·위험등급·수수료를 "
                  "설명받지 못했거나 사실과 다른 설명을 들은 경우. 설명의무 위반, 적합성·적정성 원칙 위반",
    "보이스피싱·전자금융사기": "보이스피싱, 스미싱, 명의도용, 계좌 부정이체, 전자금융 사기 피해, "
                              "사기이용계좌 지급정지와 피해금 환급",
    "대출·금리": "대출 거절, 금리 인하 요구, 이자 계산, 연체, 근저당, 상환 관련 문의와 분쟁",
    # 제20조(불공정영업행위의 금지) · 제21조(부당권유행위 금지)의 열거를 옮긴 것
    #
    # 조문의 열거를 다 옮기면(235자) 다른 유형 설명(평균 42자)보다 다섯 배 길어져
    # 이 유형이 허브가 된다 — 금감원 사례집 정답지에서 택일 정확도가 89.7%에서
    # 83.1%로 떨어졌다. 그래서 **조문의 핵심 낱말만** 남겼다(101자).
    #
    # 이 선택의 대가는 `대출·금리`와의 혼동이 늘어나는 것인데(사례집 recall@2
    # 74.2% → 70.8%), **그 혼동은 기한 카드를 0장 잃는다** — 부당권유 쪽 카드가
    # 대출·금리 쪽을 포함하기 때문이다(deadlines.py 참조). 반대로 이 낱말들이
    # 빠지면 금융위가 '불공정영업행위' 章에 넣은 질의를 절반도 회수하지 못한다
    # (프로브 53.8%). 잃는 것이 없는 쪽과 권리 안내를 놓치는 쪽의 비교다.
    "부당권유·불공정영업": "계약 강요(꺾기), 부당한 담보·보증 요구, 제3자 연대보증 요구, "
                          "중도상환수수료 부과, 편익 요구. 단정적 판단 제공, 사실과 다른 설명, "
                          "불초청 방문·전화 권유, 거부 후 재권유",
    "예금·계좌": "예금, 적금, 계좌 개설과 해지, 수수료 관련 문의",
    "카드": "신용카드, 체크카드, 할부, 청약철회",
    "보험": "보험금 지급, 보험 계약, 상해·질병·자동차 보험 관련 분쟁",
    "투자상품": "주식, 펀드, 파생상품 등 투자 관련 일반 문의",
    "절차문의": "민원 접수 방법, 분쟁조정 신청 절차와 처리 기간, 금융분쟁조정위원회, "
               "필요 서류, 관련 분쟁조정기구 안내",
    "기타": "위 유형에 해당하지 않는 금융 관련 문의",
}

_cat_vecs: np.ndarray | None = None


@dataclass
class Classification:
    category: str
    is_dispute: bool
    confidence: Literal["high", "low"]
    # 임베딩 후보 중 채택되지 않은 쪽. 기한 카드를 "해당될 수 있음"으로 함께
    # 노출하는 데 쓴다 — recall@1과 recall@2의 차이(약 15%p)에서 카드가 사라지는
    # 것을 막는다(deadlines.compute_deadlines 설명).
    alt_category: str | None = None


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
    """민원 유형은 **임베딩이 정하고**, sLLM은 분쟁성만 판정한다.

    ## 왜 sLLM에게 유형을 고르게 하지 않는가

    예전에는 임베딩이 후보 2개를 추리고 sLLM이 그중 하나를 골랐다. 그 구조는
    EXAONE 3.5 시절 "택일 정확도 89.7%"라는 측정 위에 서 있었는데, **그 측정의
    정답지가 우리 키워드 규칙이 만든 것이었다.** 금융감독원 사례집 라벨로 다시 재니
    결론이 뒤집혔다(결정례 112건, 그중 기관 라벨 18건).

        구분                     전체 112건   기관 라벨 18건
        임베딩 1순위 (LLM 미호출)    71.4%        88.9%
        임베딩 2 → sLLM 택일         58.0%        44.4%
        임베딩 top-2 포함률(천장)     86.6%          —

    **기관 라벨에서 sLLM이 정확도를 반으로 깎는다.** 1.2B 모델에게 두 유형의 경계를
    가르는 것은 비싼 요구였고, 임베딩이 이미 1순위로 맞혀 놓은 것을 2순위로
    갈아치우는 쪽이 더 많았다.

    그래서 유형은 임베딩 1순위를 그대로 쓴다. 부수 효과가 둘 있다 —
    유형 판정이 **결정론적**이 되고(같은 민원에 매번 같은 유형), 실패할 여지가
    없어진다(JSON이 깨져도 유형은 나온다).

    ## sLLM에게 남긴 것

    `is_dispute`(분쟁성)만 묻는다. 이진 판단이라 택일과 성격이 다르고, 유형과 달리
    **임베딩으로 대신할 방법이 없다.** 다만 이 값의 정확도는 정답지가 없어 재지
    못했다 — 화면 표시에만 쓰고 흐름을 가르지 않는 이유다.

    호출이 실패하면 `is_dispute=False`, `confidence="low"`로 두되 **유형은 그대로
    나간다.** 예전에는 이 경로가 유형까지 폴백이었다.
    """
    cands = _candidates(text)
    category = cands[0]
    alt = cands[1] if len(cands) > 1 else None

    system = (
        "너는 금융 민원 분류기다. 반드시 JSON으로만 답한다. "
        '형식: {"is_dispute": true|false}. '
        "is_dispute는 금융회사와의 다툼·피해구제 요구(분쟁성)이면 true, 단순 정보 문의면 false."
    )
    prompt = f"민원 내용: {text}\n\n이것은 분쟁성 민원인가?"
    schema = {
        "type": "object",
        "properties": {"is_dispute": {"type": "boolean"}},
        "required": ["is_dispute"],
    }
    try:
        data = json.loads(llm.generate(system, prompt, timeout=60, schema=schema))
        return Classification(
            category=category, is_dispute=bool(data.get("is_dispute", False)),
            confidence="high", alt_category=alt,
        )
    except (json.JSONDecodeError, ValueError, TypeError, LLMUnavailable):
        return Classification(
            category=category, is_dispute=False, confidence="low", alt_category=alt,
        )
