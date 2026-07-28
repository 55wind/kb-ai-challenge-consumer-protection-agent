import pytest
from fastapi.testclient import TestClient

from app.graph import runtime
from app.graph.context_graph import ContextGraph, Node
from app.server import db
from app.server.api import AppContext, create_app
from app.server.security import reset_admin_token_for_tests

TEST_TOKEN = "graph-test-token"


class StubLLM:
    model = "stub"

    def generate(self, *a, **k):
        return ""


def _test_graph() -> ContextGraph:
    g = ContextGraph()
    g.add_node(Node(id="cat:불완전판매", type="category", label="불완전판매"))
    g.add_node(Node(id="case-1", type="case", label="사모펀드 사례"))
    g.add_node(Node(id="law-19", type="law", label="금소법 제19조"))
    g.add_edge("case-1", "cites", "law-19",
               {"method": "regex_citation", "source_id": "case-1", "evidence": "제19조에 따라"})
    g.add_edge("case-1", "categorized_as", "cat:불완전판매", {"method": "corpus_label"})
    return g


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", TEST_TOKEN)
    reset_admin_token_for_tests()
    g = _test_graph()
    monkeypatch.setattr(runtime, "_graph", g)
    monkeypatch.setattr(runtime, "_loaded", True)
    ctx = AppContext(llm=StubLLM(), conn=db.connect(tmp_path / "g.db"))
    yield TestClient(create_app(ctx)), g
    reset_admin_token_for_tests()


def _h():
    return {"X-Admin-Token": TEST_TOKEN}


def test_graph_stats(client):
    c, _ = client
    stats = c.get("/api/admin/graph/stats", headers=_h()).json()
    assert stats["nodes"] == 3 and stats["edges"] == 2
    assert stats["edges_by_rel"]["cites"] == 1


def test_graph_requires_admin(client):
    c, _ = client
    assert c.get("/api/admin/graph/stats").status_code == 401


def test_graph_search_and_neighbors(client):
    c, _ = client
    nodes = c.get("/api/admin/graph/search", params={"q": "사모펀드"}, headers=_h()).json()["nodes"]
    assert nodes[0]["id"] == "case-1"
    nb = c.get("/api/admin/graph/neighbors", params={"id": "case-1"}, headers=_h()).json()
    rels = {n["edge"]["rel"] for n in nb["neighbors"]}
    assert rels == {"cites", "categorized_as"}
    assert all("method" in n["edge"]["provenance"] for n in nb["neighbors"])


def test_edge_toggle_curation(client):
    c, g = client
    eid = next(e.id for e in g.edges.values() if e.rel == "cites")
    r = c.post(f"/api/admin/graph/edge/{eid}/toggle", headers=_h(), json={"reason": "오추출"})
    assert r.json()["disabled"] is True
    # stats에 반영 + 이웃 조회에는 disabled 플래그로 표시
    stats = c.get("/api/admin/graph/stats", headers=_h()).json()
    assert stats["disabled_edges"] == 1
    nb = c.get("/api/admin/graph/neighbors", params={"id": "case-1"}, headers=_h()).json()
    flag = next(n for n in nb["neighbors"] if n["edge"]["id"] == eid)
    assert flag["disabled"] is True
    # 재토글 → 복원
    assert c.post(f"/api/admin/graph/edge/{eid}/toggle", headers=_h()).json()["disabled"] is False


def test_consumer_answer_has_no_evidence_graph(client, monkeypatch):
    """소비자 답변에는 근거 지도를 싣지 않는다.

    그래프는 담당자 콘솔의 감사·큐레이션 도구다. 소비자 화면에 provenance까지
    흘려보내면 쓰이지도 않는 데이터가 브라우저로 나간다.
    """
    from app.agent import orchestrator
    from app.agent.answerer import Answer, Citation
    from app.agent.classifier import Classification
    from app.agent.orchestrator import SessionState, handle_message

    monkeypatch.setattr(orchestrator.classifier, "classify",
                        lambda t, llm: Classification("불완전판매", False, "high"))
    monkeypatch.setattr(orchestrator, "retrieve", lambda kind, q, **kw: ["hit"])
    monkeypatch.setattr(orchestrator.answerer, "answer_events",
                        lambda q, h, llm: iter([("answer", Answer(text="답[1]",
                                                 citations=[Citation("law-19", "제19조", "s", "u", "q")]))]))
    events = list(handle_message(SessionState(), "설명의무 위반", StubLLM()))
    assert all(e.type != "evidence_graph" for e in events)
    # 답변·근거 카드는 그대로 나가야 한다
    assert any(e.type == "answer" for e in events)
    assert any(e.type == "citations" for e in events)


def test_admin_subgraph_returns_graph_for_visualization(client):
    """관리자 그래프 시각화용 서브그래프 — 노드·엣지와 비활성 플래그를 함께 준다."""
    c, _ = client
    sub = c.get("/api/admin/graph/subgraph", params={"id": "law-19"}, headers=_h()).json()
    assert sub["focus"] == "law-19"
    ids = {n["id"] for n in sub["nodes"]}
    assert "law-19" in ids and "case-1" in ids  # 조문을 인용한 실제 사례가 1-hop으로 연결
    assert sub["edges"], "엣지가 없으면 그릴 것이 없다"
    assert all("disabled" in e for e in sub["edges"])
    assert all("provenance" in e for e in sub["edges"])

    # id 없이 호출하면 유형 노드 기준 개관
    overview = c.get("/api/admin/graph/subgraph", headers=_h()).json()
    assert overview["focus"] is None
    assert overview["nodes"] and overview["edges"]
