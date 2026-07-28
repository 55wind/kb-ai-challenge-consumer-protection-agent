import numpy as np
import pytest

from app.agent import classifier
from app.agent.classifier import CATEGORIES, Classification, classify


class StubLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, system, prompt, json_mode=False, timeout=120):
        return self.response


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch):
    names = list(CATEGORIES)

    def fake_embed(texts):
        # 카테고리 설명 임베딩: 단위 벡터를 카테고리마다 다른 축에 배치
        if len(texts) == len(names):
            return np.eye(len(names), dtype=np.float32)
        # 질의: '피싱' 포함 시 보이스피싱 축, 아니면 불완전판매 축
        axis = names.index("보이스피싱·전자금융사기") if "피싱" in texts[0] else 0
        v = np.zeros((1, len(names)), dtype=np.float32)
        v[0, axis] = 1.0
        return v

    monkeypatch.setattr(classifier, "_cat_vecs", None)
    monkeypatch.setattr(classifier.embedder, "embed", fake_embed)


def test_classify_uses_llm_choice():
    llm = StubLLM('{"category": "보이스피싱·전자금융사기", "is_dispute": true}')
    result = classify("보이스피싱을 당해서 돈이 빠져나갔어요", llm)
    assert result == Classification("보이스피싱·전자금융사기", True, "high")


def test_classify_falls_back_on_bad_json():
    llm = StubLLM("죄송합니다, JSON이 아닙니다")
    result = classify("펀드 설명을 못 들었어요", llm)
    assert result.confidence == "low"
    assert result.category in CATEGORIES


def test_classify_falls_back_on_unknown_category():
    llm = StubLLM('{"category": "우주여행", "is_dispute": false}')
    result = classify("펀드 설명을 못 들었어요", llm)
    assert result.confidence == "low"
