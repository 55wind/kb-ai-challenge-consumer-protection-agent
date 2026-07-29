"""하이브리드 검색: 벡터 + BM25를 RRF로 결합, 관련도 임계값 게이트."""
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.corpus.loader import load_corpus
from app.search import embedder
from app.search.indexer import HybridIndex, load_index, tokenize

# 결합 상수(RRF_K·BM25_WEIGHT)의 근거는 config.py 주석과
# scripts/tune_retrieval.py 실측에 있다. 모듈 상수로 굳히지 않고 호출 때 읽는다 —
# 테스트와 튜닝 스크립트가 값을 바꿔 가며 재려면 그래야 한다.


class NoEvidenceFound(Exception):
    """근거를 댈 수 없는 상태 — 추측 금지.

    두 경우를 함께 나타낸다.
    1. 검색 결과의 최고 벡터 유사도가 임계값 미만
    2. 인덱스가 아직 빌드되지 않음 (scripts/build_index.py 미실행)

    2번을 별도 예외로 두지 않는 이유: 호출부(답변·기한·검진)가 하는 일이 같다 —
    근거를 못 대므로 판정하지 않고 물러난다. 예외 종류를 늘리면 세 곳 모두에
    같은 분기를 중복해야 하고, 하나라도 빠뜨리면 500이 난다.
    운영자용 신호는 /api/health의 indexes 체크가 따로 제공한다.
    """


@dataclass
class Hit:
    record_id: str
    score: float
    record: Any


_index_cache: dict[str, HybridIndex] = {}
_record_cache: dict[str, dict[str, Any]] = {}


def _get_index(kind: str) -> HybridIndex:
    if kind not in _index_cache:
        try:
            _index_cache[kind] = load_index(kind)
        except (OSError, RuntimeError) as e:
            # 인덱스 파일 부재·손상. FAISS는 RuntimeError, pickle·json은 OSError를 낸다.
            raise NoEvidenceFound(
                f"'{kind}' 검색 인덱스를 읽을 수 없습니다. "
                f"scripts/build_index.py를 먼저 실행하세요. ({e})"
            ) from e
    return _index_cache[kind]


def _get_records(kind: str) -> dict[str, Any]:
    if kind not in _record_cache:
        _record_cache[kind] = {r.id: r for r in load_corpus(kind)}
    return _record_cache[kind]


def rrf_merge(
    vec_hits: list[tuple[str, float]],
    bm_hits: list[tuple[str, float]],
    k: int,
) -> list[tuple[str, float]]:
    """벡터·BM25 순위를 RRF로 합친다. BM25는 소수 의견으로만 반영한다."""
    rrf_k = settings.RRF_K
    scores: dict[str, float] = {}
    for rank, (rid, _) in enumerate(vec_hits):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (rid, _) in enumerate(bm_hits):
        scores[rid] = scores.get(rid, 0.0) + settings.BM25_WEIGHT / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def retrieve(kind: str, query: str, k: int = 5) -> list[Hit]:
    index = _get_index(kind)
    records = _get_records(kind)
    qv = embedder.embed([query])[0]
    vec_hits = index.search_vec(qv, k=k * 3)
    if not vec_hits or vec_hits[0][1] < settings.RELEVANCE_THRESHOLD:
        raise NoEvidenceFound(
            f"'{query[:50]}'에 대해 확인된 근거를 찾지 못했습니다 "
            f"(최고 유사도 {vec_hits[0][1]:.2f} < {settings.RELEVANCE_THRESHOLD})"
            if vec_hits
            else "인덱스가 비어 있습니다"
        )
    bm_hits = index.search_bm25(tokenize(query), k=k * 3)
    merged = rrf_merge(vec_hits, bm_hits, k)
    return [Hit(record_id=rid, score=s, record=records[rid]) for rid, s in merged]
