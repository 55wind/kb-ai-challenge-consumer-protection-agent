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


def test_category_comes_from_embedding_not_the_llm():
    """유형은 임베딩 1순위가 정한다. sLLM이 뒤집지 못한다.

    금감원 사례집 라벨로 재니 sLLM 택일이 정확도를 반으로 깎았다(18건 기준
    88.9% → 44.4%). 예전 "택일 정확도 89.7%"는 우리 키워드 규칙이 만든 정답지로
    잰 값이었다. 이 경로가 되살아나면 그 손해가 그대로 돌아온다.
    """
    # 모델이 다른 유형을 우겨도 임베딩 1순위가 이겨야 한다
    llm = StubLLM('{"is_dispute": true, "category": "카드"}')
    result = classify("보이스피싱을 당해서 돈이 빠져나갔어요", llm)
    assert result.category == "보이스피싱·전자금융사기"
    assert result.is_dispute is True
    assert result.confidence == "high"
    # 채택되지 않은 후보가 남아야 한다 — 기한 카드를 "해당될 수 있음"으로 함께
    # 노출하는 데 쓴다(deadlines.compute_deadlines 설명).
    assert result.alt_category and result.alt_category != result.category


def test_classification_is_deterministic_for_the_same_input():
    """같은 민원에는 같은 유형. sLLM을 빼서 얻은 성질이라 고정한다."""
    llm = StubLLM('{"is_dispute": false}')
    got = {classify("펀드 설명을 못 들었어요", llm).category for _ in range(5)}
    assert len(got) == 1, got


def test_category_survives_llm_failure():
    """sLLM이 죽어도 유형은 나간다. 예전에는 이 경로가 유형까지 폴백이었다."""
    llm = StubLLM("죄송합니다, JSON이 아닙니다")
    result = classify("펀드 설명을 못 들었어요", llm)
    assert result.category in CATEGORIES
    assert result.confidence == "low"
    assert result.is_dispute is False


def test_llm_is_only_asked_about_dispute():
    """프롬프트가 유형 선택을 묻지 않는다 — 물으면 그 답을 쓰고 싶어진다."""
    from app.agent import classifier as clf

    seen = {}

    class Spy:
        def generate(self, system, prompt, **kw):
            seen["prompt"] = prompt
            seen["schema"] = kw.get("schema")
            return '{"is_dispute": false}'

    monkeypatch_free = clf.classify("보험금을 안 준대요", Spy())
    assert monkeypatch_free.category in CATEGORIES
    assert "분쟁성" in seen["prompt"]
    assert set(seen["schema"]["properties"]) == {"is_dispute"}, seen["schema"]
