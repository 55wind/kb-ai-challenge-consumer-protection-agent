"""확장이 실제로 무엇을 근거에 넣는지 눈으로 확인한다.

측정에서 '정답을 건진 문항 0건'이 나왔다. 그렇다면 추가되는 20건은
쓸모 있는 근거인가 소음인가 — 답변에 들어가는 것이므로 반드시 봐야 한다.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.graph import expand
from app.graph.runtime import curated_graph
from app.search.retriever import NoEvidenceFound, _get_records, retrieve


def main() -> int:
    entries = [
        json.loads(l)
        for l in (settings.DATA_DIR / "goldenset.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()
    ]
    graph = curated_graph()
    records = _get_records("laws")
    in_corpus = dropped = 0

    for e in entries[:6]:
        if e.get("kind", "laws") != "laws":
            continue
        print(f"\n[질문] {e['query'][:50]}")
        print(f"  기대: {e['expected_ids']}")
        try:
            hits = retrieve("laws", e["query"], k=4)
        except NoEvidenceFound:
            print("  (조문 검색 실패)")
            continue
        print(f"  검색: {[h.record_id for h in hits][:3]}")
        try:
            cases = [h.record_id for h in retrieve("cases", e["query"], k=3)]
        except NoEvidenceFound:
            cases = []
        found = expand.cited_laws(graph, cases, limit=2,
                                  exclude={h.record_id for h in hits})
        for c in found:
            ok = c.law_id in records
            in_corpus if not ok else None
            mark = "본문有" if ok else "본문無(stub·버려짐)"
            print(f"  + {c.law_id}  [{mark}]")
            print(f"      인용원문: {c.evidence[:70]}")
            if ok:
                in_corpus += 1
            else:
                dropped += 1

    print(f"\n요약: 코퍼스 본문 있음 {in_corpus}건 · stub이라 버려짐 {dropped}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
