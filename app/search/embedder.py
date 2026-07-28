"""KURE-v1 임베딩. 모델은 첫 호출 때 lazy 로드하는 싱글턴.

다중 스레드 환경(SSE 스레드풀)에서 모델 1개를 공유하므로 encode를 락으로
직렬화한다 — CPU 박스에서는 병렬 encode가 이득도 없다.
"""
import threading

import numpy as np

from app.config import settings

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        import os

        # 로컬 캐시만 사용 — 허브 버전 확인으로 인한 네트워크 의존 제거.
        # (오프라인 시연장에서 DNS 실패가 스트림 전체를 죽이는 것을 방지)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(settings.EMBED_MODEL, device="cpu")
        except Exception:
            # 캐시 미존재 등으로 오프라인 로드 실패 시 1회 온라인 재시도
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            _model = SentenceTransformer(settings.EMBED_MODEL, device="cpu")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """텍스트 목록 → L2 정규화된 (n, dim) float32 배열."""
    with _lock:
        vecs = _get_model().encode(
            texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True
        )
    return np.asarray(vecs, dtype=np.float32)
