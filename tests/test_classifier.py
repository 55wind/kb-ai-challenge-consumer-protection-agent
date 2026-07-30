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
    # `product_type`도 같은 embedder 모듈을 쓴다. 그 모듈의 `embed`를 갈아끼우면
    # `product_type._vecs`가 **가짜 벡터로 채워진 채 전역에 남는다** — monkeypatch가
    # `embed`는 되돌려도 캐시는 되돌리지 못한다. 그래서 뒤에 도는 다른 테스트가
    # 가짜 벡터로 상품유형을 추정하고, 혼자 돌리면 통과하는데 전체로 돌리면
    # 실패하는 모양이 된다(실제로 그렇게 났다).
    from app.agent import product_type
    monkeypatch.setattr(product_type, "_vecs", None)


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


# ── 민원유형은 상품유형 경유로 정한다 ────────────────────────────────
def test_categories_are_routed_through_product_type(monkeypatch):
    """1순위는 상품유형이 정한다. 임베딩 1순위가 아니어도 그것이 이긴다.

    10종을 임베딩으로 직접 고르면 recall@1이 64.5%였는데, 상품유형(제3조 4종, 실측
    95.4%)에서 내려오면 92.6%다(금감원 사례집 394건). 이 경로가 끊기면 28%p가 사라진다.
    """
    from app.agent import classifier as clf
    from app.agent import product_type

    monkeypatch.setattr(product_type, "infer", lambda t, top=2: ["보장성 상품"])
    monkeypatch.setattr(clf, "_candidates", lambda t, top=2: ["투자상품", "기타", "카드"])
    assert clf.rank_categories("보험금을 안 준대요")[0] == "보험"


def test_product_route_does_not_erase_process_inquiries(monkeypatch):
    """`절차문의`는 상품유형으로 덮지 않는다.

    사례집 정답지에 `절차문의`가 0건이라 **그 정답지로는 이 손해가 보이지 않는다.**
    폐기된 「결정례 중심 벡터」가 정확히 그 자리에서 무너졌다. 감독당국 편철 프로브에서
    top-1이 38.5%까지 떨어지는 것을 보고 넣은 방어다.
    """
    from app.agent import classifier as clf
    from app.agent import product_type

    monkeypatch.setattr(product_type, "infer", lambda t, top=2: ["대출성 상품"])
    monkeypatch.setattr(clf, "_candidates", lambda t, top=2: ["절차문의", "대출·금리", "기타"])
    assert clf.rank_categories("분쟁조정 신청은 어떻게 하나요?")[0] == "절차문의"


def test_embedding_order_survives_in_the_tail(monkeypatch):
    """2순위는 임베딩 판단을 그대로 잇는다.

    상품유형 2순위로 채우면 프로브 top-2가 92.3% → 76.9%로 떨어졌다. 2순위는
    기한 카드를 「해당될 수 있음」으로 띄우는 데 쓰이므로 좁히면 카드가 사라진다.
    """
    from app.agent import classifier as clf
    from app.agent import product_type

    monkeypatch.setattr(product_type, "infer", lambda t, top=2: ["보장성 상품"])
    monkeypatch.setattr(clf, "_candidates", lambda t, top=2: ["부당권유·불공정영업", "기타", "카드"])
    assert clf.rank_categories("보험 문제", top=2) == ["보험", "부당권유·불공정영업"]


def test_violation_words_beat_the_product_axis(monkeypatch):
    """조문이 행위를 직접 금지한 낱말은 상품유형보다 앞선다(제20·21조)."""
    from app.agent import classifier as clf
    from app.agent import product_type

    monkeypatch.setattr(product_type, "infer", lambda t, top=2: ["대출성 상품"])
    monkeypatch.setattr(clf, "_candidates", lambda t, top=2: ["대출·금리", "기타", "카드"])
    got = clf.rank_categories("대출해주면서 제3자 연대보증을 요구했습니다")
    assert got[0] == "부당권유·불공정영업"
