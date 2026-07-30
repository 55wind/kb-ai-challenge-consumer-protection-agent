"""민원유형 10종 분류를 **기관 라벨**로 잰다.

## 왜 별도 스크립트인가

`eval_model.py`의 분류 측정은 `cases.jsonl`의 `category`를 정답으로 쓴다. 그런데 그
값은 84%가 우리 키워드 규칙의 추정이다(`category_source`). 우리 규칙으로 우리 분류기를
채점하면 점수가 부풀 수밖에 없다 — 이 프로젝트가 같은 실수를 네 번 했다.

이 스크립트는 **금감원 「금융분쟁조정 사례집」의 章 라벨 394건**만 쓴다(기관 라벨 100%).
그리고 **감독당국 편철 프로브 13건을 홀드아웃**으로 함께 잰다 — 사례집에는
`절차문의`·`기타`가 0건이라, 그 정답지만 보면 그 두 유형이 무너져도 보이지 않는다.
「결정례 중심 벡터」와 「상품유형 경유(방어 전)」가 정확히 그 자리에서 붕괴했다.

## 왜 top-2도 재는가

2순위(`Classification.alt_category`)는 버려지지 않는다. 그 유형의 법정 기한 카드를
「해당될 수 있음」으로 함께 띄운다(`deadlines.compute_deadlines`). 그래서 2순위가
맞는지가 **사용자가 기한을 받는지**를 좌우한다.

재현: python scripts/eval_category.py
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

from app.agent import classifier  # noqa: E402
from app.config import settings  # noqa: E402

# 기준. 실측(사례집 394건)은 top-1 92.6% · top-2 95.7%다.
TARGET_TOP2 = 0.95


def wilson(hits: int, n: int) -> str:
    if n == 0:
        return "[-, -]"
    z, p = 1.96, hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return f"[{(c - s) / d:.1%}, {(c + s) / d:.1%}]"


def load(name: str) -> list[tuple[str, str]]:
    path = settings.DATA_DIR / name
    if not path.exists():
        return []
    return [(r["query"], r["category"])
            for r in (json.loads(x) for x in path.open(encoding="utf-8"))]


def score(rows: list[tuple[str, str]], tag: str) -> tuple[float, float]:
    n = len(rows)
    if not n:
        print(f"{tag}: 문항 없음")
        return (0.0, 0.0)
    ranked = [classifier.rank_categories(q, top=2) for q, _ in rows]
    h1 = sum(r[0] == g for r, (_, g) in zip(ranked, rows))
    h2 = sum(g in r for r, (_, g) in zip(ranked, rows))
    print(f"{tag} (n={n})")
    print(f"   분포 {dict(Counter(g for _, g in rows))}")
    print(f"   top-1 {h1}/{n} = {h1/n:6.1%}   95%CI {wilson(h1, n)}")
    print(f"   top-2 {h2}/{n} = {h2/n:6.1%}   95%CI {wilson(h2, n)}   "
          f"← 2순위 기한 카드가 여기에 걸린다")
    wrong = Counter((g, r[0]) for r, (_, g) in zip(ranked, rows) if r[0] != g)
    if wrong:
        print("   top-1 실패 상위:")
        for (gold, got), c in wrong.most_common(5):
            print(f"      {c:4d}  정답 {gold} → 예측 {got}")
    print()
    return (h1 / n, h2 / n)


def main() -> int:
    print("민원유형 10종 분류 — 기관 라벨 기준\n")
    top1, top2 = score(load("classifier_goldset.jsonl"),
                       "① 금감원 사례집 章 라벨 (공백 복원)")
    score(load("classifier_probe.jsonl"),
          "② 감독당국 편철 프로브 — **홀드아웃**")

    print("─" * 66)
    print("①의 한계: 입력이 **사례 제목**이다. 실사용 구어체와 문체가 다르다.")
    print("        그리고 `절차문의`·`기타`가 0건이라 그 두 유형은 ①이 재지 못한다.")
    print("②의 한계: 13문항뿐이라 신뢰구간이 넓다. 정확도를 주장하는 지표가 아니라")
    print("        **붕괴 경보**로 쓴다 — 상품유형 경유가 여기서 38.5%까지 떨어졌다.")
    print()
    print(f"기준 top-2 >= {TARGET_TOP2:.0%}  →  {'통과' if top2 >= TARGET_TOP2 else '미달'}"
          f"  (top-1 {top1:.1%})")
    return 0 if top2 >= TARGET_TOP2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
