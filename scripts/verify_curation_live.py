"""큐레이션이 운영 서비스에 실제로 반영되는지 종단 확인.

담당자가 관리 인스턴스에서 엣지를 끄면, 고객 인스턴스의 답변 근거에서
그 연결이 빠져야 한다. 예전에는 아무 일도 일어나지 않았다 —
관리 콘솔이 하지 않는 일을 한다고 표시하고 있었다.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.config import settings
from app.graph import expand
from app.graph.runtime import curated_graph
from app.search.retriever import retrieve
from app.server import db

QUERY = "고령인 아버지가 구조를 이해 못 하는 상품에 가입하셨어요"


def evidence_ids() -> list[str]:
    """고객 경로가 실제로 쓰는 확장 근거 (TTL 캐시를 태운 그대로)."""
    graph = curated_graph()
    cases = [h.record_id for h in retrieve("cases", QUERY, k=3)]
    return [c.law_id for c in expand.cited_laws(
        graph, cases, limit=settings.GRAPH_EXPANSION_MAX)]


def main() -> int:
    settings.GRAPH_CURATION_TTL = 0.0  # 지연 없이 확인
    graph = curated_graph()
    if graph is None:
        print("그래프가 없습니다.")
        return 1

    before = evidence_ids()
    print(f"질문: {QUERY}")
    print(f"\n1) 큐레이션 전 확장 근거: {[b.replace('ref:', '') for b in before]}")
    if not before:
        print("확장 근거가 없어 검증할 수 없습니다.")
        return 1

    # 담당자가 첫 번째 근거의 인용 엣지를 '오추출'로 끈다
    target = before[0]
    cases = [h.record_id for h in retrieve("cases", QUERY, k=3)]
    from app.graph.context_graph import edge_id
    eid = next(
        edge_id(c, "cites", target)
        for c in cases
        if edge_id(c, "cites", target) in graph.edges
    )
    conn = db.connect()
    db.set_edge_curation(conn, eid, True, "라이브 검증")
    print(f"\n2) 담당자가 엣지 비활성화: {eid}  ({target.replace('ref:', '')})")

    after = evidence_ids()
    print(f"\n3) 큐레이션 후 확장 근거: {[a.replace('ref:', '') for a in after]}")

    ok = target not in after
    print(f"\n{'통과' if ok else '실패'}: 끈 근거가 답변 경로에서 "
          f"{'빠졌습니다' if ok else '여전히 쓰입니다'}")

    # 원상 복구 — 검증이 데이터를 남기지 않게
    db.set_edge_curation(conn, eid, False, "라이브 검증 복구")
    restored = evidence_ids()
    print(f"4) 복원 후: {[r.replace('ref:', '') for r in restored]}")
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
