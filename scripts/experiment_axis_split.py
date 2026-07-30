"""유형 체계를 두 축(상품유형 × 위반유형)으로 가르면 분류 정확도가 오르는가.

## 왜 이 실험을 하나

현행 `CATEGORIES` 10종은 **두 축이 섞여 있다.** `보험`·`카드`·`예금·계좌`·
`투자상품`·`대출·금리`는 상품유형이고, `불완전판매`·`부당권유·불공정영업`·
`보이스피싱·전자금융사기`는 위반유형이다. 하나의 사건은 둘 다에 해당하므로
단일 라벨로 채점하면 **정답이 둘인데 하나만 맞다고 채점**한다.

실제로 오분류 155건 중 127건(81.9%)이 축을 건너뛴 것이었다
(`정답 보험 → 예측 불완전판매` 29건 등).

## 축을 우리가 정하지 않는다 — 조문과 금감원 편철이 정한다

상품축은 **금융소비자보호법 제3조**가 4종으로 열거한다. 우리가 나눈 것이 아니다.

    제3조(금융상품의 유형) … 1. 예금성 2. 대출성 3. 투자성 4. 보장성
    다만, 개별 금융상품이 둘 이상에 해당하는 속성이 있는 경우에는
    해당 상품유형에 각각 속하는 것으로 본다.

마지막 문장이 중요하다 — **중복 귀속을 조문이 명시적으로 허용한다.** 단일 라벨이
법에도 맞지 않았던 것이다. 그리고 제3조제2호는 **신용카드를 대출성에 넣는다**.
우리 `카드` 유형은 대출성의 부분집합이었다.

정답지 라벨은 금융감독원 「금융분쟁조정 사례집」의 편철에서 온다. 확인해 보니
사례집도 이미 두 축이다 — **章이 위반유형, 節이 상품유형**을 가리킨다.

    章 설명의무위반및부당권유  /  節 ELS및ELF        → 위반=설명의무, 상품=투자성
    章 신용카드등             /  節 신용카드의도용    → 상품=대출성(제3조2호)
    章 담보책임              /  節 포괄근저당…       → 상품=대출성

## 누출을 피한다

유형 설명에 금감원 章·節 이름을 넣으면 점수가 부풀어 오른다 — 節 이름은 금감원이
그 아래 사례들을 요약해 붙인 것이라 사례 제목과 겹치는 것이 당연하다. 이 프로젝트는
같은 실수를 두 번 했다(결정례 키워드 라벨, 프로브 자체 라벨). 그래서 **채택 후보의
설명은 조문 어휘만** 쓴다. 節 어휘를 넣은 변형도 함께 재지만 오염된 값으로 적는다.

재현: python scripts/experiment_axis_split.py
"""
import argparse
import collections
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.agent.classifier import CATEGORIES, _candidates  # noqa: E402
from app.config import settings  # noqa: E402
from app.search import embedder  # noqa: E402

# ── 상품축: 금소법 제3조 4종 ──────────────────────────────────────
# 설명은 제3조 각 호가 지시하는 제2조제1호 각 목과 시행령 제3조의 열거를 옮긴 것이다.
# 소비자 구어체 낱말(펀드·주식 등)을 우리가 보태지 않는다 — 보태면 그 낱말을 고른
# 근거가 우리 감각이 되고, 나중에 무엇이 점수를 만들었는지 알 수 없다.
PRODUCT_STATUTE: dict[str, str] = {
    "예금성": "예금, 예탁금. 「은행법」·「상호저축은행법」에 따른 예금, 「신용협동조합법」에 "
              "따른 예탁금, 「중소기업은행법」·「한국산업은행법」에 따른 예금. 금융소비자로부터 "
              "금전을 받고 장래에 그 금전과 그에 따른 이자 등의 대가를 지급하기로 하는 금융상품",
    "대출성": "대출, 신용카드, 시설대여, 연불판매, 할부금융. 「은행법」·「상호저축은행법」에 따른 "
              "대출, 「여신전문금융업법」에 따른 신용카드·시설대여·연불판매·할부금융, 「대부업법」에 "
              "따른 대부, 연계대출, 소액후불결제. 금융소비자에게 금전 또는 그 밖의 재산적 가치가 "
              "있는 것을 제공하고 장래에 그 금전등 및 그에 따른 이자 등의 대가를 받기로 하는 금융상품",
    "투자성": "금융투자상품. 「자본시장과 금융투자업에 관한 법률」에 따른 금융투자상품, 연계투자, "
              "신탁계약, 투자일임계약",
    "보장성": "보험상품. 「보험업법」에 따른 보험상품, 「신용협동조합법」에 따른 공제",
}

# 節 어휘를 보탠 변형 — 정답지와 같은 출처라 **오염된 값**으로만 적는다
PRODUCT_LEAKY: dict[str, str] = {
    k: v + extra
    for (k, v), extra in zip(
        PRODUCT_STATUTE.items(),
        [
            ". 예금 지급 및 해지, 지급정지·상계 및 착오송금, 계좌 개설 및 입금, 금리 적용, 수표대금 지급",
            ". 대출거래, 수수료 환급, 담보책임, 근저당 피담보채무 범위, 제3자 담보제공, 보증책임, "
            "연대보증, 보증의사 미확인, 신용카드 도용 및 위·변조 사용, 제3자 부당사용",
            ". 펀드 환매, ELS 및 ELF, 주식형·채권형·혼합형펀드, 부동산펀드, 선물·옵션, "
            "매매주문 착오, 반대매매, 임의매매",
            ". 질병·상해보험, 자동차보험, 암 진단, 보증 및 배상책임보험, 계약전 알릴의무 위반 및 "
            "계약해지, 재해 인정 여부, 장해 및 수술 해당 여부, 법률 및 약관 해석",
        ],
    )
}

# ── 정답지 → 상품축 매핑 ──────────────────────────────────────────
# 사례마다 판단하지 않는다. 우리 라벨 6종에 대해 **한 번** 정하고 근거를 적는다.
# 이것이 키워드 규칙과 다른 점이다(사례별 재판단 없음 → 재현·검증 가능).
LABEL_TO_PRODUCT: dict[str, tuple[str, str]] = {
    "보험":      ("보장성", "제3조4호 — 「보험업법」에 따른 보험상품"),
    "예금·계좌":  ("예금성", "제3조1호 — 章이 전부 「예금등수신거래」"),
    "대출·금리":  ("대출성", "제3조2호 — 章이 대출거래·담보책임·보증책임. 節에 시설대여(리스) 포함"),
    "카드":      ("대출성", "제3조2호가 신용카드를 대출성에 열거. 章이 전부 「신용카드등」"),
    "투자상품":   ("투자성", "제3조3호 — 章이 매매주문관련·임의매매(증권 매매)"),
    "불완전판매":  ("투자성", "제3조3호 — 節이 전부 펀드·ELS/ELF·주식형펀드·부동산펀드·선물옵션"),
}

PRODUCT_AXIS = set(PRODUCT_STATUTE)
VIOLATION_AXIS = {"불완전판매", "부당권유·불공정영업", "보이스피싱·전자금융사기"}


def rank(vecs: np.ndarray, names: list[str], query_vec: np.ndarray, top: int) -> list[str]:
    order = np.argsort(vecs @ query_vec)[::-1][:top]
    return [names[i] for i in order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldset", default=str(settings.DATA_DIR / "classifier_goldset.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.goldset, encoding="utf-8")]
    unmapped = {r["category"] for r in rows} - set(LABEL_TO_PRODUCT)
    if unmapped:
        print(f"매핑이 없는 라벨: {sorted(unmapped)} — 실험을 멈춘다", file=sys.stderr)
        return 1
    for r in rows:
        r["product"] = LABEL_TO_PRODUCT[r["category"]][0]

    n = len(rows)
    print(f"표본 {n}건 · 정답지 {Path(args.goldset).name} · 임베딩 {settings.EMBED_MODEL}")
    print("(sLLM 미호출 — 재는 것은 후보 회수, 즉 택일의 천장이다)")
    print()
    print("상품축 정답 분포:")
    dist = collections.Counter(r["product"] for r in rows)
    for k, v in dist.most_common():
        print(f"  {v:4d}  {k:<5} {v/n:6.1%}")
    major = dist.most_common(1)[0]
    print(f"  다수결 기준선(항상 {major[0]}): {major[1]/n:.1%}")
    print()

    # ── 현행 10종 단일 라벨 ────────────────────────────────────
    queries = [r["query"] for r in rows]
    qvecs = embedder.embed(queries)
    cur_names = list(CATEGORIES)
    cur_vecs = embedder.embed([f"{k}: {v}" for k, v in CATEGORIES.items()])
    cur1 = cur2 = 0
    for r, qv in zip(rows, qvecs):
        p = rank(cur_vecs, cur_names, qv, 2)
        cur1 += p[0] == r["category"]
        cur2 += r["category"] in p
    print(f"현행 10종 단일 라벨   recall@1 {cur1/n:6.1%}   recall@2 {cur2/n:6.1%}")
    print()

    # ── 상품축 4종 ─────────────────────────────────────────────
    results = {}
    for tag, table in (("조문 어휘만 (채택 후보)", PRODUCT_STATUTE),
                       ("조문 + 금감원 節 어휘 (오염)", PRODUCT_LEAKY)):
        names = list(table)
        vecs = embedder.embed([f"{k} 상품: {v}" for k, v in table.items()])
        h1 = h2 = 0
        err = collections.Counter()
        for r, qv in zip(rows, qvecs):
            p = rank(vecs, names, qv, 2)
            h1 += p[0] == r["product"]
            h2 += r["product"] in p
            if p[0] != r["product"]:
                err[(r["product"], p[0])] += 1
        results[tag] = (h1 / n, h2 / n, err)
        se = (h1 / n * (1 - h1 / n) / n) ** 0.5
        print(f"상품축 4종 · {tag}")
        print(f"   recall@1 {h1/n:6.1%}  (±{1.96*se:.1%})   recall@2 {h2/n:6.1%}")
        for (g, p), c in err.most_common(5):
            print(f"      {c:4d}  정답 {g} → 예측 {p}")
        print()

    a1 = results["조문 어휘만 (채택 후보)"][0]
    print("─" * 62)
    print(f"현행 10종 {cur1/n:.1%}  →  상품축 4종(조문) {a1:.1%}   "
          f"{'+' if a1 >= cur1/n else ''}{(a1 - cur1/n)*100:.1f}%p")
    print("주의: 두 값은 과제가 다르다(10지선다 vs 4지선다). 같은 잣대의 비교가 아니라")
    print("      '기한 룰을 정하는 데 필요한 판단'을 각각 얼마나 맞히는지의 비교다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
