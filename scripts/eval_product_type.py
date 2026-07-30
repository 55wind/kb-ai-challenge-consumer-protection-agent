"""상품유형 판정을 구어체 정답지로 잰다.

이 판정이 청약철회 기간을 고른다(제46조①: 보장성 15일·투자성 7일·대출성 14일).
`product_type.infer`는 **하나로 좁히지 않고 상위 2개**를 돌려주므로, 재야 할 것은
top-1 정확도가 아니라 **top-2 포함률**이다 — 두 카드를 함께 보여주기 때문이다.

정답지: `scripts/build_product_goldset.py`가 조문 열거로 만든 것.
비교용으로 사례집 정답지(문어체 사례 제목)도 함께 잰다 — 문체 차이를 보이려고.
"""
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.agent import product_type  # noqa: E402
from app.config import settings  # noqa: E402

# 사례집 라벨(6종) → 제3조 상품유형. `build_classifier_goldset` 대응표와 같다.
CASE_LABEL_TO_PRODUCT = {
    "보험": "보장성 상품", "예금·계좌": "예금성 상품", "대출·금리": "대출성 상품",
    "카드": "대출성 상품", "투자상품": "투자성 상품", "불완전판매": "투자성 상품",
}


def wilson(hits: int, n: int) -> tuple[float, float]:
    """이항 비율의 95% 신뢰구간(윌슨). 100%에서도 구간이 1이 되지 않는다 —
    작은 표본에서 '100%'를 확정으로 읽지 않게 하려고 이 구간을 함께 적는다."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def score(rows: list[tuple[str, str]], tag: str) -> None:
    n = len(rows)
    if not n:
        print(f"{tag}: 문항 없음")
        return
    t1 = t2 = 0
    misses = []
    for q, truth in rows:
        pred = product_type.infer(q, top=2)
        if pred[0] == truth:
            t1 += 1
        if truth in pred:
            t2 += 1
        else:
            misses.append((q, truth, pred))
    lo1, hi1 = wilson(t1, n)
    lo2, hi2 = wilson(t2, n)
    print(f"{tag} (n={n})")
    print(f"   top-1 {t1}/{n} = {t1/n:6.1%}   95%CI [{lo1:.1%}, {hi1:.1%}]")
    print(f"   top-2 {t2}/{n} = {t2/n:6.1%}   95%CI [{lo2:.1%}, {hi2:.1%}]   ← 실제 쓰는 값")
    print(f"   정답 분포: {dict(Counter(t for _, t in rows))}")
    if misses:
        print("   top-2에서도 놓친 문항:")
        for q, truth, pred in misses[:6]:
            print(f"      [{truth}] {q[:60]}")
            print(f"         → 추정 {pred}")
    print()


def main() -> int:
    path = settings.DATA_DIR / "product_goldset.jsonl"
    if not path.exists():
        print("먼저 scripts/build_product_goldset.py 를 실행하세요", file=sys.stderr)
        return 1
    colloquial = [(r["query"], r["product_type"])
                  for r in (json.loads(x) for x in path.open(encoding="utf-8"))]
    score(colloquial, "① 구어체 (조문 적용 라벨)")

    cg = settings.DATA_DIR / "classifier_goldset.jsonl"
    if cg.exists():
        literary = [(r["query"], CASE_LABEL_TO_PRODUCT[r["category"]])
                    for r in (json.loads(x) for x in cg.open(encoding="utf-8"))
                    if r["category"] in CASE_LABEL_TO_PRODUCT]
        score(literary, "② 사례집 사례 제목 (문어체 · 본문에서 공백 복원)")

    print("─" * 62)
    print("두 정답지의 문체가 다르다. ①이 실사용에 가깝고 ②는 표본이 크다.")
    print()
    print("①의 한계: 상품 이름이 **문자로 나온** 문항만 담긴다. 사용자가 상품을")
    print("        말하지 않았을 때의 정확도는 이 지표가 재지 않는다.")
    print()
    print("②의 한계: `product_type.PRODUCT_TYPES`의 유형 설명에 금감원 사례집의")
    print("        章·節 어휘가 들어 있고(계약 전 알릴 의무, 임의매매, 포괄근저당 …),")
    print("        ②의 라벨도 같은 章·節에서 왔다. **어휘 출처가 같아 값이 부푼다.**")
    print("        조문 어휘만으로 재면 top-2가 92.6%였다")
    print("        (`scripts/experiment_axis_split.py`). ②는 상한으로 읽는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
