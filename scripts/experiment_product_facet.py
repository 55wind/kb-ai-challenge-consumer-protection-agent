"""분류를 **한 축**으로 좁히면 정확도가 어디까지 가는지 잰다.

**왜 이 실험이 필요한가.**

현재 `CATEGORIES` 10종은 서로 다른 축을 섞어 놓았다.

    상품·권역 축 : 보험 · 카드 · 예금·계좌 · 투자상품
    위반 축      : 불완전판매 · 부당권유·불공정영업
    수단 축      : 보이스피싱·전자금융사기
    메타         : 절차문의 · 기타

**보험 불완전판매 사건은 `보험`이면서 동시에 `불완전판매`다.** 배타적이지 않은
것을 하나만 고르라고 하면, 모델이 아무리 좋아도 정답률에 천장이 생긴다. 실제로
금감원 라벨 기준 62.5%이고, 후보 회수율 천장이 70.8%다.

**금소법 제3조가 이 문제를 이미 인정한다.** 상품유형을 예금성·대출성·투자성·
보장성 넷으로 구분하면서 단서를 달았다 — *"개별 금융상품이 둘 이상에 해당하는
속성이 있는 경우에는 해당 상품유형에 각각 속하는 것으로 본다."* 조문이 다중
소속을 전제하는데 우리 설계는 단일 라벨이다.

그래서 **축을 하나로 좁힌 경우**를 잰다. 여기서 재는 축은 제3조의 상품유형이고,
정답은 금감원 사례집의 편·장 구조에서 나온다(우리 판단이 아니다).

    보험편 9개 장                                    → 보장성
    대출거래 · 담보책임 · 보증책임 · 신용카드등        → 대출성  (제3조 2호가 신용카드를 포함)
    예금등수신거래                                   → 예금성
    금융투자상품판매 · 설명의무위반및부당권유
      · 매매주문관련 · 임의매매                       → 투자성

이 실험이 답하는 것: **"95%가 지금 지표에서 안 나오는 것이 모델 탓인가, 과제
설정 탓인가."**
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.llm.ollama_client import OllamaClient  # noqa: E402
from app.search import embedder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# 금소법 제3조의 상품유형. 설명도 조문에서 가져온다.
PRODUCT_TYPES = {
    "보장성 상품": "보험상품. 생명보험·손해보험·자동차보험, 보험금 지급, 보험계약 해지, "
                  "계약 전 알릴 의무, 재해·장해·질병 인정 여부",
    "투자성 상품": "금융투자상품. 펀드·신탁·ELS·파생결합증권·주식·선물옵션, "
                  "설명의무 위반과 부당권유, 매매주문, 임의매매",
    "대출성 상품": "대출, 신용카드·시설대여·연불판매·할부금융. "
                  "담보와 보증, 근저당, 연체와 상환, 중도상환수수료",
    "예금성 상품": "예금과 적금. 계좌 개설과 해지, 금리 적용, 예금 지급, "
                  "지급정지·상계·착오송금",
}

# 금감원 사례집 장 → 제3조 상품유형. 편·장 구조가 곧 라벨이다.
CHAPTER_TO_PRODUCT = {
    # 보험편 — 전권이 보장성 상품
    "계약전알릴의무위반및계약해지등": "보장성 상품",
    "재해등인정여부": "보장성 상품",
    "암진단등": "보장성 상품",
    "장해및수술해당여부": "보장성 상품",
    "법률및약관해석등": "보장성 상품",
    "자동차보험": "보장성 상품",
    "질병․상해등보험": "보장성 상품",
    "보증및배상책임보험": "보장성 상품",
    "기타": "보장성 상품",
    # 은행·중소서민금융·금융투자편
    "대출거래": "대출성 상품",
    "담보책임": "대출성 상품",
    "보증책임": "대출성 상품",
    "신용카드등": "대출성 상품",      # 제3조 2호가 신용카드를 대출성에 넣는다
    "예금등수신거래": "예금성 상품",
    "금융투자상품판매": "투자성 상품",
    "설명의무위반및부당권유": "투자성 상품",
    "매매주문관련": "투자성 상품",
    "임의매매": "투자성 상품",
    # 전산장애·사무처리 오류는 상품유형과 무관하다 — 제외
    "전산장애,사무처리오류등": None,
}

NAMES = list(PRODUCT_TYPES)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def load(limit: int) -> list[tuple[str, str]]:
    rows = [json.loads(x) for x in
            (ROOT / "data/fss_case_index.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for e in rows:
        p = CHAPTER_TO_PRODUCT.get(e["chapter"])
        if p:
            out.append((e["title"], p))
    import random
    random.Random(20260730).shuffle(out)
    return out[:limit]


def classify(text: str, cands: list[str], llm) -> str:
    system = ('너는 금융상품 유형 분류기다. 반드시 JSON으로만 답한다. '
              '형식: {"product_type": "<후보 중 하나>"}.')
    listed = "\n".join(f"- {c}: {PRODUCT_TYPES[c]}" for c in cands)
    prompt = f"분쟁 사안: {text}\n\n후보 유형:\n{listed}\n\n어느 상품유형인가?"
    try:
        got = json.loads(llm.generate(system, prompt, json_mode=True, timeout=60)).get("product_type", "")
        return got if got in PRODUCT_TYPES else cands[0]
    except Exception:  # noqa: BLE001
        return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    data = load(args.limit)
    print(f"금감원 사례집 {len(data)}건 · 정답은 편·장 구조 · 유형 {len(NAMES)}종 (금소법 제3조)")
    print("분포:", dict(collections.Counter(g for _, g in data)), "\n")

    mat = _unit(embedder.embed([f"{n}: {d}" for n, d in PRODUCT_TYPES.items()]))
    ranks = []
    for text, gold in data:
        qv = _unit(embedder.embed([text])[0])
        order = [NAMES[i] for i in np.argsort(mat @ qv)[::-1]]
        ranks.append(order.index(gold) + 1)
    n = len(data)
    print("① 임베딩 단독 (후보 추림)")
    for k in (1, 2, 3):
        hit = sum(1 for r in ranks if r <= k)
        print(f"  recall@{k}  {hit}/{n}  ({hit / n:.1%})")

    if args.no_llm:
        return 0

    print("\n② sLLM 확정")
    llm = OllamaClient()
    for k in (2, 4):
        hit = 0
        conf: collections.Counter = collections.Counter()
        for text, gold in data:
            qv = _unit(embedder.embed([text])[0])
            order = [NAMES[i] for i in np.argsort(mat @ qv)[::-1]]
            got = classify(text, order[:k], llm)
            hit += got == gold
            if got != gold:
                conf[(gold, got)] += 1
        label = "후보 2개" if k == 2 else "4종 전부 제시"
        print(f"  {label:<12} {hit}/{n}  ({hit / n:.1%})   "
              f"주요 오분류: {[f'{a}→{b}' for (a, b), _ in conf.most_common(3)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
