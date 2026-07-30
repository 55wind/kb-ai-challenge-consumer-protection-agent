"""같은 프롬프트를 CPU와 GPU에서 각각 돌려 생성 속도를 비교한다.

문서에 "CPU 실측"이라고 적으려면 실제로 CPU에서 재야 한다. Ollama는 GPU가
보이면 알아서 쓰기 때문에, 그냥 재면 GPU 수치가 나오고도 CPU라고 적게 된다.
요청 단위로 num_gpu=0을 줘서 서버 설정을 건드리지 않고 CPU로 강제한다.

Ollama가 응답에 실어주는 나노초 카운터를 쓴다 — 벽시계로 재면 HTTP 왕복과
모델 적재가 섞인다.
"""
import io
import statistics
import sys

import httpx

from app.config import settings

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:11434"
# 리터럴로 박으면 모델을 교체한 뒤에도 옛 모델을 재고 그 수치를 문서에 적게 된다
MODEL = settings.GEN_MODEL
PROMPT = ("다음 조문을 근거로 답하세요.\n"
          "[1] 예금자보호법 제32조 — 예금보험공사는 예금자에게 5천만원 한도로 보험금을 지급한다.\n\n"
          "질문: 예금자보호 한도는 얼마인가요?")
ROUNDS = 3


def run(num_gpu: int | None) -> dict:
    opts = {"temperature": 0.2, "num_ctx": 4096}
    if num_gpu is not None:
        opts["num_gpu"] = num_gpu
    payload = {"model": MODEL, "prompt": PROMPT, "stream": False, "options": opts}
    r = httpx.post(f"{BASE}/api/generate", json=payload, timeout=900)
    r.raise_for_status()
    d = r.json()
    ns = 1_000_000_000
    return {
        "첫 토큰까지": (d["prompt_eval_duration"] + d.get("load_duration", 0)) / ns,
        "생성": d["eval_duration"] / ns,
        "전체": d["total_duration"] / ns,
        "토큰수": d["eval_count"],
        "초당토큰": d["eval_count"] / (d["eval_duration"] / ns),
    }


def bench(label: str, num_gpu: int | None) -> dict:
    print(f"  {label} 측정 중 ({ROUNDS}회)...", flush=True)
    run(num_gpu)  # 워밍업 — 모델 적재를 측정에서 뺀다
    rows = [run(num_gpu) for _ in range(ROUNDS)]
    return {k: statistics.median(r[k] for r in rows) for k in rows[0]}


def main() -> int:
    try:
        httpx.get(f"{BASE}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Ollama에 연결할 수 없습니다: {e}")
        return 1

    print(f"모델 {MODEL} — 같은 프롬프트, num_gpu만 다르게\n")
    gpu = bench("GPU", None)      # 기본값 = Ollama가 알아서 GPU 사용
    cpu = bench("CPU", 0)         # 레이어를 GPU에 하나도 안 올림

    print(f"\n{'항목':<14}{'GPU':>12}{'CPU':>12}{'배수':>10}")
    print("-" * 48)
    for k in ("첫 토큰까지", "생성", "전체"):
        ratio = cpu[k] / gpu[k] if gpu[k] else 0
        print(f"{k:<14}{gpu[k]:>11.2f}초{cpu[k]:>11.2f}초{ratio:>9.1f}배")
    print(f"{'초당 토큰':<14}{gpu['초당토큰']:>11.1f}{cpu['초당토큰']:>12.1f}"
          f"{gpu['초당토큰'] / cpu['초당토큰']:>9.1f}배")
    print("-" * 48)
    print(f"생성 토큰 {int(gpu['토큰수'])}개 기준")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
