"""분류 정확도 80%가 무엇으로 이루어져 있는지 분해한다.

**왜 분해해야 하는가.** "80%"는 손댈 곳을 알려주지 않는다. 분류는 두 단계다.

    민원 텍스트 ──▶ ① 임베딩으로 후보 2개 추림 ──▶ ② sLLM이 둘 중 택일

정답이 ①에서 이미 빠졌다면 ②는 아무리 좋은 모델을 써도 맞출 수 없다. 그런데
지금까지 잰 것은 최종 정확도뿐이라 **어느 단계가 병목인지 모르는 상태**였다.

여기서 재는 것은 셋이다.

  1. **후보 회수율(recall@k)** — 정답 유형이 후보 안에 들어오는 비율.
     이것이 ②의 천장이다. recall@2가 85%면 최종 정확도는 절대 85%를 못 넘는다.
  2. **택일 정확도** — 정답이 후보에 있을 때 sLLM이 그것을 고르는 비율.
     여기가 낮으면 모델·프롬프트 문제다.
  3. **분류 체계 불일치** — 모델이 고른 유형이 정답지에 아예 없는 경우.
     `CATEGORIES`는 10개인데 결정례 라벨은 5개뿐이라, 나머지 5개는 고르는 순간
     오답으로 집계된다. 이것은 모델의 잘못이 아니라 **측정의 성질**이다.

사용:
    python scripts/diagnose_classifier.py --cases 60
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.agent.classifier import CATEGORIES, _candidates, classify  # noqa: E402
from app.llm.ollama_client import OllamaClient  # noqa: E402
from scripts.eval_model import _cases  # noqa: E402

BAR = "─" * 66


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--no-llm", action="store_true",
                    help="후보 회수율만 잰다 (sLLM 호출 없음 — 빠르다)")
    args = ap.parse_args()

    data = _cases(args.cases)
    gold_labels = {g for _, g in data}
    unreachable = sorted(set(CATEGORIES) - gold_labels)

    print(f"표본 {len(data)}건")
    print(f"정답지 라벨 {len(gold_labels)}종: {sorted(gold_labels)}")
    print(f"분류기 유형 {len(CATEGORIES)}종 — 정답지에 없는 {len(unreachable)}종: {unreachable}")
    print("  (이 유형을 고르면 무조건 오답으로 집계된다. 모델 잘못이 아니라 측정의 성질이다.)")

    # ── ① 후보 회수율 ──────────────────────────────────────────────
    print(f"\n{BAR}\n① 후보 회수율 — sLLM이 고를 수 있는 천장")
    ranks = []
    for text, gold in data:
        order = _candidates(text, top=len(CATEGORIES))
        ranks.append(order.index(gold) + 1 if gold in order else 99)

    n = len(ranks)
    for k in (1, 2, 3, 4, 5):
        hit = sum(1 for r in ranks if r <= k)
        mark = "  ← 현재 설정" if k == 2 else ""
        print(f"  recall@{k}  {hit}/{n}  ({hit / n:.1%}){mark}")

    # ── ② 택일 정확도 ──────────────────────────────────────────────
    if args.no_llm:
        return 0

    print(f"\n{BAR}\n② 택일 정확도 — 정답이 후보에 있을 때 sLLM이 고르는가")
    llm = OllamaClient()
    reachable = wrong_pick = fell_out = 0
    out_of_taxonomy: collections.Counter = collections.Counter()
    genuine: collections.Counter = collections.Counter()

    for (text, gold), rank in zip(data, ranks):
        got = classify(text, llm).category
        if rank <= 2:
            reachable += 1
            if got != gold:
                wrong_pick += 1
                genuine[(gold, got)] += 1
        else:
            fell_out += 1
            if got != gold:
                (out_of_taxonomy if got in set(CATEGORIES) - gold_labels
                 else genuine)[(gold, got)] += 1

    ok = reachable - wrong_pick
    print(f"  정답이 후보 안에 있던 경우: {reachable}건")
    print(f"    맞춤 {ok}건 · 틀림 {wrong_pick}건 → 택일 정확도 "
          f"{ok / reachable:.1%}" if reachable else "")
    print(f"  정답이 후보 밖으로 빠진 경우: {fell_out}건 (sLLM은 맞출 수 없었다)")

    print(f"\n{BAR}\n③ 오답의 성격")
    total_wrong = wrong_pick + fell_out - (reachable - wrong_pick - ok)
    oot = sum(out_of_taxonomy.values())
    print(f"  분류 체계 밖 유형을 고름: {oot}건 "
          f"({', '.join(f'{g}→{p}' for (g, p), _ in out_of_taxonomy.most_common(4))})")
    print(f"  체계 안에서 틀림:       {sum(genuine.values())}건 "
          f"({', '.join(f'{g}→{p}' for (g, p), _ in genuine.most_common(4))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
