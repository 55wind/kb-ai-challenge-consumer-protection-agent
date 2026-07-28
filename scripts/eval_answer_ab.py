"""그래프 확장 켬/끔 답변을 나란히 비교 (실제 sLLM 호출).

측정 지표가 아니라 사람이 읽고 판단하기 위한 도구다. 근거를 추가했는데
답변이 나빠지면 확장을 끄는 편이 낫다 — 근거 수가 아니라 답변이 목적이다.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.agent import answerer, orchestrator
from app.config import settings
from app.llm.ollama_client import OllamaClient
from app.search.retriever import retrieve

QUERIES = sys.argv[1:] or [
    "고령인 아버지가 구조를 이해 못 하는 상품에 가입하셨어요",
    "투자 위험을 알려주지 않아 손해를 봤는데 배상받을 수 있나요",
]


def run(query: str, llm, expansion: bool) -> None:
    settings.GRAPH_EXPANSION = expansion
    hits = retrieve("laws", query, k=4)
    hits = orchestrator._expand_by_citation(query, hits[:3])
    try:
        hits = hits + retrieve("kb_products", query, k=2)
    except Exception:  # noqa: BLE001
        pass
    ans = answerer.answer_with_retry(query, hits, llm)
    tag = "확장 ON " if expansion else "확장 OFF"
    print(f"\n── {tag} ──  근거 {len(hits)}건")
    for i, h in enumerate(hits, 1):
        star = "*" if getattr(h.record, "source", "").find("결정례") >= 0 else " "
        print(f"  {star}[{i}] {answerer._doc_title(h.record)[:52]}")
    print(f"\n{ans.text.strip()[:700]}")
    print(f"\n  인용한 근거: {[c.record_id for c in ans.citations]}")


def main() -> int:
    llm = OllamaClient()
    for q in QUERIES:
        print("\n" + "=" * 68)
        print(f"질문: {q}")
        print("=" * 68)
        for flag in (False, True):
            try:
                run(q, llm, flag)
            except Exception as e:  # noqa: BLE001
                print(f"  실패: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
