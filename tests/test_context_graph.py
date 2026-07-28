from app.graph.citations import extract_law_citations
from app.graph.context_graph import ContextGraph, Node


def _graph() -> ContextGraph:
    g = ContextGraph()
    g.add_node(Node(id="cat:불완전판매", type="category", label="불완전판매"))
    g.add_node(Node(id="case-1", type="case", label="사례1"))
    g.add_node(Node(id="law-19", type="law", label="금소법 제19조"))
    g.add_node(Node(id="proc-1", type="procedure", label="분쟁조정"))
    g.add_edge("case-1", "cites", "law-19", {"method": "regex_citation", "evidence": "제19조에 따라"})
    g.add_edge("case-1", "categorized_as", "cat:불완전판매", {"method": "corpus_label"})
    g.add_edge("cat:불완전판매", "remedied_by", "proc-1", {"method": "static_map"})
    return g


def test_citation_extraction_basic():
    text = "피신청인은 금융소비자 보호에 관한 법률 제19조에 따른 설명의무를 위반하였다"
    cites = extract_law_citations(text)
    assert cites[0][0] == "금융소비자 보호에 관한 법률-제19조"
    assert "제19조" in cites[0][1]  # evidence 스니펫


def test_citation_extraction_alias_and_decree():
    text = "금소법 제47조 및 통신사기피해환급법 시행령 제3조를 적용한다. 자본시장법 제85조는 별론."
    ids = [c[0] for c in extract_law_citations(text)]
    assert "금융소비자 보호에 관한 법률-제47조" in ids
    assert "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법 시행령-제3조" in ids
    assert not any("자본시장법" in i for i in ids)  # 코퍼스 밖 법령은 버림


def test_citation_article_sub_number():
    ids = [c[0] for c in extract_law_citations("금소법 제16조의2에 따라")]
    assert ids == ["금융소비자 보호에 관한 법률-제16조의2"]


def test_edge_requires_existing_nodes():
    g = _graph()
    assert g.add_edge("case-1", "cites", "ghost-node", {}) is None


def test_subgraph_hops_and_disable():
    g = _graph()
    sub = g.subgraph(["case-1"], hops=1)
    ids = {n["id"] for n in sub["nodes"]}
    assert ids == {"case-1", "law-19", "cat:불완전판매"}

    sub2 = g.subgraph(["case-1"], hops=2)
    assert "proc-1" in {n["id"] for n in sub2["nodes"]}

    # 엣지 비활성화 → 확장에서 제외
    eid = next(e for e in g.edges.values() if e.rel == "cites").id
    g.disabled.add(eid)
    sub3 = g.subgraph(["case-1"], hops=1)
    assert "law-19" not in {n["id"] for n in sub3["nodes"]}


def test_stats_and_search():
    g = _graph()
    s = g.stats()
    assert s["nodes"] == 4 and s["edges"] == 3
    assert s["edges_by_rel"]["cites"] == 1
    assert g.search("제19조")[0].id == "law-19"


def test_save_load_roundtrip(tmp_path):
    g = _graph()
    path = g.save(tmp_path / "graph.json")
    g2 = ContextGraph.load(path)
    assert set(g2.nodes) == set(g.nodes)
    assert set(g2.edges) == set(g.edges)
    assert g2.edges[next(iter(g2.edges))].provenance is not None
