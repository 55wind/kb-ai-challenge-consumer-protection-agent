import numpy as np
import pytest

from app.agent import classifier
from app.agent.classifier import CATEGORIES, Classification, classify


class StubLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, system, prompt, json_mode=False, timeout=120, schema=None):
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


def test_prompt_includes_candidate_descriptions(monkeypatch):
    """후보를 이름만 주지 말고 설명과 함께 준다.

    '불완전판매'가 설명의무 위반을 가리킨다는 것은 이 분류 체계의 약속이지
    낱말에서 자명하게 나오는 뜻이 아니다. 이름만 주면 모델이 추측해야 한다.
    결정례 60건 실측에서 설명을 붙이자 80.0% → 83.3%였다.
    """
    from app.agent import classifier as clf

    seen = {}

    class Spy:
        def generate(self, system, prompt, **kw):
            seen["prompt"] = prompt
            return '{"category": "보험", "is_dispute": false}'

    monkeypatch.setattr(clf, "_candidates", lambda t, top=2: ["보험", "카드"])
    clf.classify("보험금을 안 준대요", Spy())
    assert clf.CATEGORIES["보험"] in seen["prompt"], "후보 설명이 프롬프트에 없다"
    assert clf.CATEGORIES["카드"] in seen["prompt"], "두 번째 후보 설명이 빠졌다"
