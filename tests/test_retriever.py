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


def test_bm25_only_docs_lose_to_vector_top3():
    """BM25에만 있고 벡터에는 없는 문서가 벡터 3위를 이기면 안 된다.

    실제로 났던 문제를 그대로 옮긴 것이다. 소비자는 "상담해준 사람이 은행
    직원인지 대리점인지 밝히지 않았어요"라고 쓰지 "고지의무"라고 쓰지 않는다.
    겹치는 낱말이 없으니 BM25는 "금융상품·판매·대리·중개"가 잔뜩 들어간
    **등록요건 조문**을 물어 오는데, 예전 설정(K=60, 1:1)에서는 그것이
    1/61 = 0.0164점을 받아 벡터 3위(1/63 = 0.0159)를 눌렀다. 순위 차이보다
    "두 목록에 나왔느냐"가 더 크게 작용한 것이다.
    """
    vec = [("noise1", 0.9), ("noise2", 0.9), ("정답", 0.8)]
    bm = [("등록요건_조문", 20.0), ("또다른_잡음", 10.0)]
    merged = [rid for rid, _ in rrf_merge(vec, bm, k=5)]
    assert merged.index("정답") < merged.index("등록요건_조문")


def test_bm25_boost_is_bounded():
    """BM25가 문서를 끌어올릴 수 있는 폭에 상한이 있어야 한다.

    BM25를 아주 죽이지는 않는다 — 조문 번호나 고유명사가 든 질문은 BM25가
    잡아낸다. 대신 폭을 묶는다. BM25 1위라는 최대 가산을 받아도 벡터 순위를
    다섯 계단 넘게 뛰어넘지는 못해야, 벡터가 제대로 찾은 근거가 상위 5건
    밖으로 밀려나지 않는다.
    """
    from app.config import settings

    boost = settings.BM25_WEIGHT / (settings.RRF_K + 1)  # BM25 1위가 주는 최대 가산
    rank5 = 1.0 / (settings.RRF_K + 5)
    # 벡터 5위가 최대 가산을 받아도 벡터 1위 자리까지는 못 간다
    assert rank5 + boost < 1.0 / (settings.RRF_K + 1) * 1.35, (
        f"BM25 가산({boost:.4f})이 너무 커서 벡터 순위를 뒤집는다"
    )


def test_rrf_k_keeps_rank_meaningful():
    """1위와 5위의 점수 차가 20% 이상 나야 순위가 실제로 순위 구실을 한다.

    RRF 원 논문의 K=60은 수백~수천 건짜리 목록을 합칠 때의 값이다. 이 서비스는
    후보를 15개만 뽑으므로 60을 그대로 쓰면 1위와 5위가 6% 차이밖에 안 난다.
    """
    from app.config import settings

    first = 1.0 / (settings.RRF_K + 1)
    fifth = 1.0 / (settings.RRF_K + 5)
    assert (first - fifth) / first > 0.2, (
        f"RRF_K={settings.RRF_K}에서는 1위와 5위가 사실상 같은 점수다"
    )


def test_goldenset_ids_exist():
    """골든셋 정답 ID가 코퍼스에 실재하는지.

    코퍼스가 갱신되면 결정례 번호가 재배정된다(`fss-제2021-9호-…`가
    `fss-제2023-1호-…`로 바뀐 적이 있다). 정답 ID가 사라져도 평가는 조용히
    돌고, 그 문항은 영원히 맞출 수 없는 문항이 된다. 점수가 떨어지면 검색을
    의심하게 되는데 원인은 골든셋에 있다.
    """
    import json

    from app.config import settings
    from app.corpus.loader import load_corpus

    entries = [
        json.loads(line)
        for line in (settings.DATA_DIR / "goldenset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) >= 50, "골든셋이 50문항 미만이면 표본이 너무 작다"
    known: dict[str, set[str]] = {}
    missing = []
    for e in entries:
        kind = e["kind"]
        if kind not in known:
            known[kind] = {r.id for r in load_corpus(kind)}
        missing += [(e["query"][:30], i) for i in e["expected_ids"] if i not in known[kind]]
    assert not missing, f"코퍼스에 없는 정답 ID: {missing}"
