import numpy as np
import pytest

from app.corpus.schema import LawChunk
from app.search import retriever
from app.search.retriever import Hit, NoEvidenceFound, rrf_merge


def test_rrf_merge_prefers_agreement():
    vec = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm = [("b", 12.0), ("d", 3.0)]
    merged = rrf_merge(vec, bm, k=4)
    assert merged[0][0] == "b"  # 양쪽 모두에 등장


def _stub_env(monkeypatch, top_sim: float):
    rec = LawChunk.validated(
        id="fx-1", law_name="[TEST-FIXTURE] 법", article="제1조(테스트)",
        text="본문", source="[TEST-FIXTURE] pytest", source_url="about:test",
    )

    class FakeIndex:
        def search_vec(self, qv, k):
            return [("fx-1", top_sim)]

        def search_bm25(self, tokens, k):
            return [("fx-1", 5.0)]

    monkeypatch.setattr(retriever, "_get_index", lambda kind: FakeIndex())
    monkeypatch.setattr(retriever, "_get_records", lambda kind: {"fx-1": rec})
    monkeypatch.setattr(retriever.embedder, "embed", lambda texts: np.zeros((1, 4), dtype=np.float32))
    monkeypatch.setattr(retriever, "tokenize", lambda q: ["토큰"])


def test_retrieve_returns_hits(monkeypatch):
    _stub_env(monkeypatch, top_sim=0.8)
    hits = retriever.retrieve("laws", "설명의무", k=3)
    assert isinstance(hits[0], Hit)
    assert hits[0].record_id == "fx-1"


def test_retrieve_gates_low_relevance(monkeypatch):
    _stub_env(monkeypatch, top_sim=0.1)
    with pytest.raises(NoEvidenceFound):
        retriever.retrieve("laws", "오늘 날씨 어때", k=3)


def test_missing_index_becomes_no_evidence(monkeypatch, tmp_path):
    """인덱스 미빌드 상태에서 FileNotFoundError가 새어나가면 안 된다.

    zip을 갓 푼 환경(scripts/build_index.py 미실행)에서 기한 계산·검진이
    500으로 죽는 것을 막는다 — 근거만 빠지고 기능은 계속 동작해야 한다.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "INDEX_DIR", tmp_path)
    retriever._index_cache.clear()
    with pytest.raises(NoEvidenceFound, match="build_index"):
        retriever.retrieve("laws", "설명의무", k=3)
    retriever._index_cache.clear()


def test_deadlines_survive_missing_index(monkeypatch, tmp_path):
    """인덱스가 없어도 골든타임 카드는 나온다 (근거 링크만 빠진다)."""
    from datetime import date

    from app.agent.deadlines import compute_deadlines
    from app.config import settings

    monkeypatch.setattr(settings, "INDEX_DIR", tmp_path)
    retriever._index_cache.clear()
    items = compute_deadlines("불완전판매", "2026년 7월 1일", today=date(2026, 7, 23))
    assert items, "기한 항목이 비면 안 된다"
    assert all(i.law_citation is None for i in items)
    assert any(i.computed_deadline for i in items)
    retriever._index_cache.clear()
