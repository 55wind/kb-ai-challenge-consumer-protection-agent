"""응답 지연을 단계별로 쪼개 잰다.

어디를 고쳐야 체감이 줄어드는지 알기 위한 것이다. 콜드 로드는 워밍업으로
분리한다 — 첫 호출의 모델 적재 20초를 매 질의 비용으로 착각하면 엉뚱한 곳을
최적화하게 된다.
"""
import io
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

QUERIES = [
    "예금자보호 한도는 얼마인가요?",
    "펀드 손실이 났는데 설명을 제대로 못 들었어요",
    "보이스피싱을 당했습니다 어떻게 해야 하나요",
]


def main() -> int:
    from app.agent import classifier
    from app.graph import runtime as graph_runtime
    from app.llm.ollama_client import OllamaClient
    from app.search.embedder import embed
    from app.search.retriever import retrieve

    llm = OllamaClient()

    print("워밍업 (임베딩 모델·그래프·인덱스 적재)...")
    t = time.perf_counter()
    embed(["워밍업"])
    graph_runtime.get_graph()
    try:
        retrieve("laws", QUERIES[0], k=2)  # 관련도 게이트를 통과하는 질의여야 한다
    except Exception:
        pass
    print(f"  콜드 로드 {time.perf_counter() - t:.1f}초 — 이후 질의에는 붙지 않는다\n")

    marks: dict[str, list[float]] = {"임베딩": [], "검색(임베딩 포함)": [],
                                     "분류(LLM)": [], "생성(LLM)": []}

    for q in QUERIES:
        t = time.perf_counter()
        embed([q])
        marks["임베딩"].append(time.perf_counter() - t)

        t = time.perf_counter()
        hits = retrieve("laws", q, k=4)
        marks["검색(임베딩 포함)"].append(time.perf_counter() - t)

        t = time.perf_counter()
        classifier.classify(q, llm)
        marks["분류(LLM)"].append(time.perf_counter() - t)

        t = time.perf_counter()
        from app.agent import answerer
        try:
            answerer.answer(q, hits, llm)
        except Exception:
            pass
        marks["생성(LLM)"].append(time.perf_counter() - t)

    print(f"{'단계':<20}{'중앙값':>10}{'최대':>10}")
    print("-" * 40)
    total = 0.0
    for name, xs in marks.items():
        med = statistics.median(xs)
        print(f"{name:<20}{med:>9.2f}초{max(xs):>9.2f}초")
        if name != "임베딩":  # 검색에 이미 포함되므로 이중 계산 방지
            total += med
    print("-" * 40)
    print(f"{'질의당 합계':<20}{total:>9.2f}초")
    print("\n임베딩은 검색 시간 안에 포함된 값이다. 검색에서 임베딩이 차지하는 비중이")
    print("크면 GPU 임베딩이 효과가 있고, 작으면 LLM 쪽이 병목이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
