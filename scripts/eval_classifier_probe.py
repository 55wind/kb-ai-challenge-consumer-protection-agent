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

**라벨은 우리가 붙이지 않는다.** 처음에는 골든셋 질의를 가져와 우리가 유형을
붙였는데, 그것은 분류 정답지에서 걷어낸 문제와 같은 문제였다 — 우리가 만든
라벨로 우리 분류기를 재는 것. 그래서 `build_classifier_probe.py`가
**감독당국 문서의 소재**로 라벨을 정한다.

    부당권유·불공정영업  금융위 「금소법 FAQ(3차)」의 章
                        「Ⅰ. 불공정영업행위 관련」·「Ⅱ. 특정 영업행위의 '권유'여부…」
    절차문의            금감원 민원·분쟁조정 절차 안내 페이지 자체

**비워 둔 유형이 있다.** `보이스피싱·전자금융사기`는 사례집에 해당 장이 없고,
가까운 절(「신용카드의 도용 및 위·변조 사용」)은 **우리 분류 체계가 겹쳐서**
정답을 정할 수 없다 — `카드` 설명에도 "카드 부정사용"이 있다. 억지로 밀어 넣으면
또 우리 판단이 라벨이 되므로 비운다. 이것은 자료 부족이 아니라 **분류 체계의
결함**이다. `기타`는 정의상 대응 자료가 없다.

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
# 금감원 사례집 정답지(`classifier_goldset.jsonl`)가 이미 검증하는 유형.
# 이 프로브가 지키는 것은 **사례집이 닿지 않는 유형**이다.
COVERED_BY_CASEBOOK = {"불완전판매", "보험", "카드", "대출·금리", "예금·계좌", "투자상품"}
# 하한은 **회귀 감시선이고 목표치가 아니다.**
#
# 조문 기반 설명으로 바꾼 뒤 측정값이 69.2%(9/13)다. 좋은 값이 아니다. 남은 4건은
# 우리 분류 체계가 실제로 겹치는 자리다 — `연대보증`은 금소법 제20조상 불공정영업
# 이지만 우리 `대출·금리`도 "근저당·상환"을 담고 있고, `보험 갱신이 권유인가`는
# 금융위가 '권유 여부 판단' 章에 넣었지만 소비자 눈에는 보험 문의다. 후보 2개로는
# 이런 2지 충돌에서 항상 금융위 쪽 답을 낼 수 없다.
#
# 그래서 하한을 0.6에 둔다. 지금보다 나빠지면 걸리게 하되, 이 선을 넘는 것이
# 목표라고 말하지 않는다. 목표는 겹침을 없애는 **분류 체계 정비**이고 본선 범위다.
FLOOR = 0.6


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
        key = "사례집이 검증함" if p["category"] in COVERED_BY_CASEBOOK else "사례집 밖 유형"
        groups[key].append(ok)
        if not ok:
            misses.append((p["category"], p["query"], cands))

    total = [x for v in groups.values() for x in v]
    print(f"프로브 {len(total)}건 (라벨: 감독당국 문서의 소재) · 후보 2개 회수율 "
          f"{sum(total)}/{len(total)} ({sum(total) / len(total):.1%})")
    for name, results in sorted(groups.items()):
        print(f"  {name:<10} {sum(results)}/{len(results)} ({sum(results) / len(results):.0%})")
    for cat, q, cands in misses:
        print(f"  [놓침] ({cat}) {q[:44]} → {cands}")

    caseless = groups["사례집 밖 유형"]
    rate = sum(caseless) / len(caseless) if caseless else 1.0
    if rate < FLOOR:
        print(f"\n사례집 밖 유형의 회수율이 {rate:.0%}로 하한 {FLOOR:.0%} 미만입니다. "
              f"유형 벡터가 특정 유형에 쏠렸는지, 유형 설명이 조문의 열거를 "
              f"빠뜨렸는지 확인하세요.")
        return 1
    print(f"\n통과 — 회귀 없음 (하한 {FLOOR:.0%}). "
          f"이 수치는 목표치가 아니다 — 모듈 주석의 설명을 보라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
