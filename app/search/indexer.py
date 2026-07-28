"""FAISS(IndexFlatIP) + BM25(Kiwi 토큰) 하이브리드 인덱스 빌드/로드."""
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from app.config import settings
from app.search.embedder import embed

_kiwi = None


def tokenize(text: str) -> list[str]:
    """명사·동사·형용사 어간 토큰 (BM25용)."""
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return [
        t.form
        for t in _kiwi.tokenize(text)
        if t.tag.startswith(("NN", "VV", "VA", "SL", "SN"))
    ]


def _paths(kind: str, index_dir: Path) -> dict[str, Path]:
    return {
        "faiss": index_dir / f"{kind}.faiss",
        "bm25": index_dir / f"{kind}.bm25.pkl",
        "ids": index_dir / f"{kind}.ids.json",
    }


def build_index(
    kind: str,
    records: list[Any],
    text_fn: Callable[[Any], str],
    index_dir: Path | None = None,
) -> None:
    index_dir = index_dir or settings.INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    p = _paths(kind, index_dir)
    texts = [text_fn(r) for r in records]
    vecs = embed(texts)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(p["faiss"]))
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with p["bm25"].open("wb") as f:
        pickle.dump(bm25, f)
    p["ids"].write_text(
        json.dumps([r.id for r in records], ensure_ascii=False), encoding="utf-8"
    )


@dataclass
class HybridIndex:
    faiss_index: Any
    bm25: BM25Okapi
    ids: list[str]

    def search_vec(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        scores, idxs = self.faiss_index.search(query_vec.reshape(1, -1), min(k, len(self.ids)))
        return [
            (self.ids[i], float(s))
            for i, s in zip(idxs[0], scores[0])
            if i >= 0
        ]

    def search_bm25(self, tokens: list[str], k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in order]


def load_index(kind: str, index_dir: Path | None = None) -> HybridIndex:
    index_dir = index_dir or settings.INDEX_DIR
    p = _paths(kind, index_dir)
    with p["bm25"].open("rb") as f:
        bm25 = pickle.load(f)
    return HybridIndex(
        faiss_index=faiss.read_index(str(p["faiss"])),
        bm25=bm25,
        ids=json.loads(p["ids"].read_text(encoding="utf-8")),
    )
