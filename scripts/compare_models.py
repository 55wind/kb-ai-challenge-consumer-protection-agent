"""생성 모델 후보를 같은 잣대로 나란히 잰다.

**왜 필요한가.** 현재 쓰는 EXAONE 3.5는 비상업 라이선스라 KB가 실서비스로
돌릴 수 없다(`10_라이선스_검토`). 그래서 "바꿀 수 있다"가 아니라 "바꾸면
이만큼 나온다"를 숫자로 둬야 한다.

재는 것은 `eval_model.py`와 같다 — **맡긴 작업의 수행률**이지 일반 벤치마크
점수가 아니다. 그쪽 함수를 그대로 가져다 쓰므로 잣대가 어긋날 일이 없다.

**측정 위생.** 모델을 바꿔 가며 잴 때 두 가지를 반드시 지킨다. 둘 다 실제로
틀린 수치를 만든 적이 있다.

  1. 모델 사이에 `ollama stop`을 넣는다. Ollama는 적재된 모델을 몇 분간 물고
     있어서, 그대로 다음 모델을 부르면 두 모델이 VRAM을 나눠 갖고 양쪽 속도가
     함께 느려진다.
  2. 두 측정을 동시에 돌리지 않는다. 같은 Ollama를 두고 경합하면 생성 2.4초짜리가
     55초로 나온다.

사용:
    python scripts/compare_models.py --cases 30 --citations 20
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.llm.ollama_client import OllamaClient  # noqa: E402
from scripts.eval_model import (  # noqa: E402
    eval_citation,
    eval_classification,
    eval_consistency,
)

BAR = "─" * 74

# (표시 이름, Ollama 태그, 라이선스, 상업 이용 가능 여부)
CANDIDATES = [
    ("EXAONE 3.5 2.4B", "exaone3.5:2.4b", "EXAONE 1.1 - NC", False),
    ("EXAONE 4.0 1.2B", "hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:Q4_K_M", "EXAONE 1.2 - NC", False),
    ("Mi:dm 2.0 Mini 2.3B", "hf.co/mykor/Midm-2.0-Mini-Instruct-gguf:Q4_K_M", "MIT", True),
]


def _unload(tag: str) -> None:
    """적재된 모델을 내린다. 실패해도 진행한다 — 이미 내려가 있으면 오류가 난다."""
    subprocess.run(["ollama", "stop", tag], capture_output=True, timeout=60)


def _loaded() -> str:
    """지금 적재된 모델. 측정 결과에 함께 남겨 오염을 사후에 알아볼 수 있게 한다."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30)
        lines = [x for x in out.stdout.splitlines()[1:] if x.strip()]
        return " / ".join(x.split()[0] for x in lines) or "(없음)"
    except Exception:  # noqa: BLE001
        return "(확인 실패)"


def measure(tag: str, cases: int, citations: int, rounds: int) -> dict:
    llm = OllamaClient(model=tag)
    t0 = time.time()

    cls = eval_classification(llm, cases)
    con = eval_consistency(llm, rounds)

    # 인용 준수는 골든셋 전체를 돌면 모델 하나당 수십 분이 걸린다. 앞에서부터
    # 잘라 쓰되 **세 모델 모두 같은 문항**을 쓴다 — 비교가 목적이므로 표본이
    # 같은 것이 표본이 큰 것보다 중요하다.
    import scripts.eval_model as em
    full = em._goldenset
    em._goldenset = lambda: full()[:citations]
    try:
        cit = eval_citation(llm)
    finally:
        em._goldenset = full

    return {
        "분류 정확도": cls["정확도"],
        "기한누락 오분류": cls["기한누락"],
        "JSON 준수율": cls["JSON 준수율"],
        "분류 일관성": con["일관성"],
        "인용 1차": cit["1차 준수율"],
        "인용 최종": cit["최종 준수율"],
        "인용 최종실패": cit["최종 실패"],
        "답변 지연(중앙값)": cit["지연_중앙값"],
        "답변 평균글자": cit["답변_평균글자"],
        "분류 지연(중앙값)": cls["지연_중앙값"],
        "_총_소요초": time.time() - t0,
        "_적재확인": _loaded(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=30, help="분류 평가 표본 수")
    ap.add_argument("--citations", type=int, default=20, help="인용 평가 골든셋 문항 수")
    ap.add_argument("--rounds", type=int, default=3, help="일관성 반복 횟수")
    ap.add_argument("--only", help="쉼표로 구분한 태그 일부만 측정")
    args = ap.parse_args()

    targets = CANDIDATES
    if args.only:
        want = [x.strip() for x in args.only.split(",")]
        targets = [c for c in CANDIDATES if any(w in c[1] for w in want)]

    print(f"분류 {args.cases}건 · 인용 {args.citations}문항 · 일관성 {args.rounds}회\n")
    results = {}
    for name, tag, lic, commercial in targets:
        for _, other_tag, _, _ in CANDIDATES:
            _unload(other_tag)          # 이전 모델을 확실히 내린다
        print(f"{BAR}\n{name}  ({lic}{'' if commercial else ' — 실서비스 불가'})")
        print(f"  태그: {tag}")
        try:
            r = measure(tag, args.cases, args.citations, args.rounds)
        except Exception as e:  # noqa: BLE001
            print(f"  측정 실패: {type(e).__name__}: {e}")
            continue
        results[name] = r
        print(f"  완료 ({r['_총_소요초']:.0f}초, 적재={r['_적재확인']})")

    if not results:
        print("측정된 모델이 없다.")
        return 1

    print(f"\n{BAR}\n비교")
    names = list(results)
    keys = [k for k in next(iter(results.values())) if not k.startswith("_")]
    print(f"{'항목':<20}" + "".join(f"{n:<24}" for n in names))
    for k in keys:
        cells = ""
        for n in names:
            v = results[n][k]
            if "율" in k or "정확도" in k or "일관성" in k or k.startswith("인용"):
                cells += f"{v:.1%}".ljust(24) if isinstance(v, float) and v <= 1 else f"{v}".ljust(24)
            elif "지연" in k:
                cells += f"{v:.2f}초".ljust(24)
            elif "글자" in k:
                cells += f"{v:.0f}자".ljust(24)
            else:
                cells += f"{v}".ljust(24)
        print(f"{k:<20}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
