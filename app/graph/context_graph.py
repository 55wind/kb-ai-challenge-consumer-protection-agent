"""컨텍스트 그래프: 트리플(주어-관계-목적어) + 엣지별 provenance.

semantica 설계 원칙 차용 — 그래프 구축에 LLM을 쓰지 않고(결정론),
모든 엣지가 검증 가능한 출처(provenance)를 가지며, 관리자는 오추출
엣지를 '비활성화만' 할 수 있다(창작·수정 불가 — 실데이터 원칙).
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import settings

NODE_TYPES = ("category", "law", "case", "procedure", "product", "rule", "deadline")


@dataclass
class Node:
    id: str
    type: str
    label: str
    meta: dict = field(default_factory=dict)


@dataclass
class Edge:
    id: str
    src: str
    rel: str
    dst: str
    provenance: dict = field(default_factory=dict)


def edge_id(src: str, rel: str, dst: str) -> str:
    return hashlib.sha1(f"{src}|{rel}|{dst}".encode()).hexdigest()[:16]


class ContextGraph:
    def __init__(self, nodes: list[Node] | None = None, edges: list[Edge] | None = None):
        self.nodes: dict[str, Node] = {n.id: n for n in (nodes or [])}
        self.edges: dict[str, Edge] = {e.id: e for e in (edges or [])}
        self.disabled: set[str] = set()
        self._out: dict[str, list[str]] = {}
        self._in: dict[str, list[str]] = {}
        for e in self.edges.values():
            self._out.setdefault(e.src, []).append(e.id)
            self._in.setdefault(e.dst, []).append(e.id)

    # ── 구축 ──
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, src: str, rel: str, dst: str, provenance: dict) -> Edge | None:
        if src not in self.nodes or dst not in self.nodes:
            return None
        eid = edge_id(src, rel, dst)
        if eid in self.edges:
            return self.edges[eid]
        e = Edge(id=eid, src=src, rel=rel, dst=dst, provenance=provenance)
        self.edges[eid] = e
        self._out.setdefault(src, []).append(eid)
        self._in.setdefault(dst, []).append(eid)
        return e

    # ── 조회 ──
    # disabled를 인자로 받는 이유: 이 그래프는 프로세스 싱글턴이라 요청마다
    # self.disabled를 바꿨다 되돌리면 스레드풀에서 실행되는 두 요청이 서로의
    # 값을 덮어써 큐레이션이 통째로 날아간다. 조회 시점의 필터는 인자로 넘긴다.
    # (None이면 인스턴스 값을 쓰고, 전부 보려면 빈 집합을 넘긴다)
    def _disabled(self, override: set[str] | None) -> set[str]:
        return self.disabled if override is None else override

    def neighbors(self, node_id: str, rel: str | None = None,
                  disabled: set[str] | None = None) -> list[tuple[Edge, Node]]:
        off = self._disabled(disabled)
        result = []
        for eid in self._out.get(node_id, []):
            e = self.edges[eid]
            if eid in off or (rel and e.rel != rel):
                continue
            result.append((e, self.nodes[e.dst]))
        for eid in self._in.get(node_id, []):
            e = self.edges[eid]
            if eid in off or (rel and e.rel != rel):
                continue
            result.append((e, self.nodes[e.src]))
        return result

    def subgraph(self, seeds: list[str], hops: int = 1, max_nodes: int = 30,
                 exclude_rels: set[str] | None = None,
                 disabled: set[str] | None = None) -> dict:
        """seeds에서 hops 단계 확장한 서브그래프 (비활성·제외 관계 필터)."""
        exclude = exclude_rels or set()
        off = self._disabled(disabled)
        keep = [s for s in seeds if s in self.nodes]
        visited = set(keep)
        frontier = list(keep)
        edges_out: dict[str, Edge] = {}
        for _ in range(hops):
            next_frontier = []
            for nid in frontier:
                for e, other in self.neighbors(nid, disabled=off):
                    if e.rel in exclude:
                        continue
                    edges_out[e.id] = e
                    if other.id not in visited and len(visited) < max_nodes:
                        visited.add(other.id)
                        next_frontier.append(other.id)
            frontier = next_frontier
        # 내부 노드 간 엣지 보강
        for e in self.edges.values():
            if e.rel in exclude:
                continue
            if e.id not in off and e.src in visited and e.dst in visited:
                edges_out[e.id] = e
        # 고립 시드 정리: 엣지에 참여하지 않는 노드는 제외
        connected = {e.src for e in edges_out.values()} | {e.dst for e in edges_out.values()}
        visited &= connected if connected else visited
        return {
            "nodes": [asdict(self.nodes[nid]) for nid in visited],
            "edges": [asdict(e) for e in edges_out.values()],
        }

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for n in self.nodes.values():
            by_type[n.type] = by_type.get(n.type, 0) + 1
        by_rel: dict[str, int] = {}
        for e in self.edges.values():
            by_rel[e.rel] = by_rel.get(e.rel, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "disabled_edges": len(self.disabled),
            "nodes_by_type": by_type,
            "edges_by_rel": by_rel,
        }

    def search(self, query: str, limit: int = 20) -> list[Node]:
        q = query.strip().lower()
        return [n for n in self.nodes.values() if q in n.label.lower() or q in n.id.lower()][:limit]

    # ── 영속화 ──
    def save(self, path: Path | None = None) -> Path:
        path = path or settings.INDEX_DIR / "graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ContextGraph":
        path = path or settings.INDEX_DIR / "graph.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            nodes=[Node(**n) for n in data["nodes"]],
            edges=[Edge(**e) for e in data["edges"]],
        )
