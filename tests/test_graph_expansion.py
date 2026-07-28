"""컨텍스트 그래프가 운영 서비스에서 실제로 작동하는지 검증.

핵심 두 가지:
1. 조문 검색이 놓친 근거를 결정례의 인용을 따라 보강한다 (검색을 연결로)
2. 담당자가 끈 엣지는 소비자 답변에서 실제로 빠진다 (큐레이션이 효력을 갖는다)

2번이 특히 중요하다. 이전에는 담당자가 엣지를 꺼도 답변은 그대로였다 —
관리 콘솔이 하지 않는 일을 한다고 표시하고 있었다.
"""
import pytest

from app.graph import expand
from app.graph.context_graph import ContextGraph, Node


def _graph() -> ContextGraph:
    g = ContextGraph()
    g.add_node(Node(id="case-1", type="case", label="사모펀드 불완전판매"))
    g.add_node(Node(id="case-2", type="case", label="특정금전신탁 불완전판매"))
    g.add_node(Node(id="law-19", type="law", label="금소법 제19조"))
    g.add_node(Node(id="law-21", type="law", label="금소법 제21조"))
    # 두 결정례가 함께 인용한 조문(law-19)이 더 앞서야 한다
    g.add_edge("case-1", "cites", "law-19",
               {"method": "regex_citation", "evidence": "제19조에 따라 설명하여야"})
    g.add_edge("case-2", "cites", "law-19",
               {"method": "regex_citation", "evidence": "제19조 설명의무"})
    g.add_edge("case-1", "cites", "law-21",
               {"method": "regex_citation", "evidence": "제21조 부당권유"})
    # 임베딩 추정 관계 — 근거 확장에 쓰이면 안 된다
    g.add_edge("case-1", "similar_to", "case-2",
               {"method": "embedding_similarity", "evidence": "cosine=0.83"})
    return g


def test_follows_citations_from_cases():
    """결정례가 원문에서 인용한 조문을 근거로 끌어온다."""
    found = expand.cited_laws(_graph(), ["case-1", "case-2"], limit=2)
    ids = [c.law_id for c in found]
    assert ids[0] == "law-19", "여러 사례가 함께 인용한 조문이 우선"
    assert set(ids) == {"law-19", "law-21"}
    assert "제19조" in found[0].evidence  # 근거 스니펫이 따라온다


def test_ignores_embedding_edges():
    """similar_to(임베딩 추정)는 근거 확장에 쓰지 않는다.

    근거로 제시하려면 출처가 원문이어야 한다. 유사도는 추정이지 사실이 아니다.
    """
    found = expand.cited_laws(_graph(), ["case-1"], limit=10)
    assert all(c.law_id.startswith("law-") for c in found)
    assert "case-2" not in [c.law_id for c in found]


def test_excludes_already_found():
    """검색이 이미 찾은 조문은 중복해서 넣지 않는다."""
    found = expand.cited_laws(_graph(), ["case-1", "case-2"], limit=5,
                              exclude={"law-19"})
    assert [c.law_id for c in found] == ["law-21"]


def test_disabled_edge_drops_from_evidence():
    """담당자가 끈 엣지는 근거 확장에서 빠진다 — 큐레이션이 효력을 갖는 지점."""
    g = _graph()
    before = {c.law_id for c in expand.cited_laws(g, ["case-1", "case-2"], limit=5)}
    assert "law-21" in before

    from app.graph.context_graph import edge_id
    g.disabled = {edge_id("case-1", "cites", "law-21")}

    after = {c.law_id for c in expand.cited_laws(g, ["case-1", "case-2"], limit=5)}
    assert "law-21" not in after, "비활성화한 엣지가 여전히 근거로 쓰인다"
    assert "law-19" in after, "다른 근거까지 사라지면 안 된다"


def test_stub_citation_becomes_usable_evidence():
    """코퍼스 밖 법령도 결정례의 인용 문장을 근거로 삼아 쓸 수 있어야 한다.

    결정례 98건은 대부분 금소법 시행(2021) 이전 사건이라 자본시장법·민법을
    인용한다. 그 법령 본문은 우리에게 없지만, 위원회가 법리를 서술한 문장은
    결정례 원문에 있다. 이걸 버리면 인용 163건 중 158건이 죽는다.
    """
    from app.graph.context_graph import Node

    g = ContextGraph()
    g.add_node(Node(id="case-1", type="case", label="사모펀드 불완전판매"))
    g.add_node(Node(id="ref:자본시장법-제47조", type="law", label="자본시장법 제47조",
                    meta={"stub": True, "source": "국가법령정보센터 자본시장법",
                          "source_url": "https://law.go.kr/..."}))
    g.add_edge("case-1", "cites", "ref:자본시장법-제47조", {
        "method": "regex_citation",
        "evidence": "금융투자업자는 일반투자자를 상대로 투자권유하는 경우에는 "
                    "금융투자상품의 내용, 투자에 따르는 위험을 이해할 수 있도록 "
                    "설명하여야 한다(자본시장법 제47조 제1항).",
    })
    found = expand.cited_laws(g, ["case-1"], limit=2)
    rec = expand.as_record(found[0], "사모펀드 불완전판매 손해배상")
    assert rec is not None
    assert "제47조" in rec.text and "설명하여야" in rec.text
    assert rec.source_url.startswith("https://law.go.kr")
    # 출처를 결정례로 밝힌다 — 법령 본문을 인용한 것이 아니다
    assert "결정례" in rec.source and "인용" in rec.source


def test_fragment_snippet_is_rejected():
    """조문 번호도 없는 조각은 근거로 쓰지 않는다.

    인용 위치 근처를 고정 폭으로 떠오므로 법리가 없는 조각이 섞인다.
    조각을 근거랍시고 LLM에 넣으면 엉뚱한 답이 나온다.
    """
    frag = expand.CitedLaw(law_id="ref:민법-제109조", cited_by="case-1",
                           evidence="익증권 매매계약 취소 관련 1) 이 사건 펀드의 일부 투자자들은")
    assert expand.as_record(frag) is None, "조문 번호 없는 조각이 근거로 통과했다"

    short = expand.CitedLaw(law_id="ref:민법-제109조", cited_by="case-1",
                            evidence="민법 제109조")
    assert expand.as_record(short) is None, "너무 짧은 인용이 통과했다"


def test_no_graph_is_noop():
    """그래프가 없으면 아무 일도 하지 않는다 — 부가 레이어이지 의존 대상이 아니다."""
    assert expand.cited_laws(None, ["case-1"]) == []
    assert expand.cited_laws(_graph(), []) == []


def test_unknown_case_id_is_skipped():
    """인덱스에는 있으나 그래프에 없는 결정례여도 죽지 않는다."""
    found = expand.cited_laws(_graph(), ["없는-사례", "case-1"], limit=5)
    assert {c.law_id for c in found} == {"law-19", "law-21"}


def test_curation_crosses_instances(tmp_path, monkeypatch):
    """관리 인스턴스의 큐레이션이 고객 인스턴스에 전달된다.

    운영에서 app(고객용)과 app-admin(관리용)은 별도 프로세스이고 공유하는 것은
    데이터 볼륨의 SQLite뿐이다. 큐레이션을 메모리로만 들고 있으면 고객 답변에는
    영원히 반영되지 않는다.
    """
    from app.config import settings
    from app.graph import runtime
    from app.server import db

    db_path = tmp_path / "shared.db"
    conn = db.connect(db_path)          # '관리 인스턴스'가 쓰는 DB
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(settings, "GRAPH_CURATION_TTL", 0.0)  # 지연 없이 확인
    monkeypatch.setattr(runtime, "_graph", _graph())
    monkeypatch.setattr(runtime, "_loaded", True)
    monkeypatch.setattr(runtime, "_curation_at", 0.0)

    assert runtime.curated_graph().disabled == set()

    from app.graph.context_graph import edge_id
    eid = edge_id("case-1", "cites", "law-21")
    db.set_edge_curation(conn, eid, True, "오추출")

    # '고객 인스턴스'는 공유 DB를 읽어 따라온다
    assert eid in runtime.curated_graph().disabled
    conn.close()


@pytest.mark.parametrize("flag", [True, False])
def test_expansion_kill_switch(monkeypatch, flag):
    """GRAPH_EXPANSION=false면 그래프 이전 동작으로 즉시 되돌아간다."""
    from app.agent import orchestrator
    from app.config import settings
    from app.search.retriever import Hit

    monkeypatch.setattr(settings, "GRAPH_EXPANSION", flag)
    called = []
    monkeypatch.setattr(orchestrator, "curated_graph",
                        lambda *a, **k: called.append(1) or None)
    base = [Hit(record_id="law-1", score=1.0, record=object())]
    assert orchestrator._expand_by_citation("질문", base) == base
    assert bool(called) is flag
