"""유형 분류의 후보 생성이 **결정례 없는 유형을 굶기지 않는지** 검사한다.

이 검사가 왜 따로 필요한가. 유형 벡터를 결정례 중심(centroid)으로 바꾸는 안을
실측했다가 폐기한 적이 있다. 결정례 정답지에서는 좋아 보였다.

    결정례 60건 recall@2 : 90.0% → 98.3%

그런데 그 정답지에는 **결정례가 있는 5개 유형만** 들어 있다. `CATEGORIES`는 10개다.
구어체 프로브로 다시 재니 이랬다.

    구어체 27건 recall@2 : 96.3% → 85.2%
      결정례 있는 유형(20건) : 20/20 → 20/20   (이득 0)
      결정례 없는 유형(7건)  :  6/7  →  3/7    (붕괴)

보험·불완전판매의 중심이 강해지면서 부당권유·절차문의 질의를 빨아들인 것이다.
**정답지가 못 보는 곳에서 손해가 났고, 정답지만 보면 개선으로 읽힌다.**

그래서 이 프로브를 남긴다. 유형 벡터 산출 방식을 손대는 사람은 결정례 정답지만
보고 판단하면 안 된다.

프로브의 한계도 적어 둔다. 질의는 골든셋(검색 평가용으로 먼저 작성된 것)에서
그대로 가져왔지만 **유형 라벨은 우리가 붙였다.** 금감원이 붙인 결정례 라벨과
달리 판단이 섞여 있다. 그래서 정확도를 주장하는 지표가 아니라 **회귀를 잡는
경보**로만 쓴다.

사용:
    python scripts/eval_classifier_probe.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.agent.classifier import CATEGORIES, _candidates  # noqa: E402
from app.config import settings  # noqa: E402

# 결정례 코퍼스에 라벨이 존재하는 유형. 나머지는 사람이 쓴 설명 벡터에만 의존한다.
WITH_CASES = {"불완전판매", "보험", "보이스피싱·전자금융사기", "카드", "대출·금리"}
FLOOR = 0.8  # 결정례 없는 유형의 recall@2 하한


def main() -> int:
    path = settings.DATA_DIR / "classifier_probe.jsonl"
    probes = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    unknown = {p["category"] for p in probes} - set(CATEGORIES)
    if unknown:
        print(f"프로브에 분류 체계 밖 유형이 있습니다: {sorted(unknown)}")
        return 2

    groups: dict[str, list[bool]] = defaultdict(list)
    misses = []
    for p in probes:
        cands = _candidates(p["query"], top=2)
        ok = p["category"] in cands
        groups["결정례 있음" if p["category"] in WITH_CASES else "결정례 없음"].append(ok)
        if not ok:
            misses.append((p["category"], p["query"], cands))

    total = [x for v in groups.values() for x in v]
    print(f"구어체 프로브 {len(total)}건 · 후보 2개 회수율 {sum(total)}/{len(total)} "
          f"({sum(total) / len(total):.1%})")
    for name, results in sorted(groups.items()):
        print(f"  {name:<10} {sum(results)}/{len(results)} ({sum(results) / len(results):.0%})")
    for cat, q, cands in misses:
        print(f"  [놓침] ({cat}) {q[:44]} → {cands}")

    caseless = groups["결정례 없음"]
    rate = sum(caseless) / len(caseless) if caseless else 1.0
    if rate < FLOOR:
        print(f"\n결정례 없는 유형의 회수율이 {rate:.0%}로 하한 {FLOOR:.0%} 미만입니다. "
              f"유형 벡터가 특정 유형에 쏠렸는지 확인하세요.")
        return 1
    print("\n통과 — 결정례 없는 유형이 밀려나지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
