"""후보 개수(top-k)를 몇으로 둘지 **금감원 라벨 정답지**로 정한다.

전에 우리 라벨(키워드 규칙)로 쟀을 때는 후보를 넓혀도 소용없어 보였다.

    우리 라벨 60건 : recall@2 90.0% → @3 91.7% → @5 93.3%   (+1.7%p, +3.3%p)

그래서 "넓히기는 답이 아니다"라고 적었다. 금감원 라벨로 다시 재니 뒤집혔다.

    금감원 라벨 120건 : recall@2 72.5% → @3 81.7% → @5 90.8%  (+9.2%p, +18.3%p)

우리 라벨이 우리가 쓴 유형 설명과 같은 어휘를 공유해서 회수율을 부풀렸던 것이다.
후보를 넓히면 천장이 크게 올라가지만 대가가 있다 — sLLM에게 주는 선택지가 늘면
택일이 어려워지고, 형식 준수도 흔들릴 수 있다(09 §4). 그래서 **천장이 아니라
최종 정확도로** 정한다.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.agent.classifier import CATEGORIES, _candidates  # noqa: E402
from app.llm.ollama_client import OllamaClient  # noqa: E402
from scripts.diagnose_classifier import _fss  # noqa: E402

BAR = "─" * 60


def classify_topk(text: str, llm, k: int) -> tuple[str, bool]:
    """현재 classify()와 같은 프롬프트로, 후보 개수만 바꿔 부른다."""
    cands = _candidates(text, top=k)
    system = (
        "너는 금융 민원 분류기다. 반드시 JSON으로만 답한다. "
        '형식: {"category": "<후보 중 하나>", "is_dispute": true|false}. '
        "is_dispute는 금융회사와의 다툼·피해구제 요구(분쟁성)이면 true, 단순 정보 문의면 false."
    )
    listed = "\n".join(f"- {c}: {CATEGORIES[c]}" for c in cands)
    prompt = f"민원 내용: {text}\n\n후보 유형:\n{listed}\n\n어느 유형인가?"
    try:
        got = json.loads(llm.generate(system, prompt, json_mode=True, timeout=60)).get("category", "")
        if got in CATEGORIES:
            return got, True
    except Exception:  # noqa: BLE001
        pass
    return cands[0], False   # 폴백 (임베딩 1순위)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--ks", default="2,3,4,5")
    args = ap.parse_args()

    data = _fss(args.limit)
    llm = OllamaClient()
    print(f"금감원 사례집 정답지 {len(data)}건 · 라벨 출처: 금융감독원\n")
    print(f"{'후보 수':<10}{'최종 정확도':<18}{'회수율(천장)':<16}{'형식 준수':<12}")
    for k in [int(x) for x in args.ks.split(",")]:
        hit = ok_fmt = reachable = 0
        for text, gold in data:
            reachable += gold in _candidates(text, top=k)
            got, fmt = classify_topk(text, llm, k)
            hit += got == gold
            ok_fmt += fmt
        n = len(data)
        print(f"{k:<10}{f'{hit}/{n} ({hit / n:.1%})':<18}"
              f"{f'{reachable / n:.1%}':<16}{f'{ok_fmt / n:.1%}':<12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
