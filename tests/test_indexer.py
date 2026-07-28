import pytest

from app.corpus.schema import LawChunk
from app.search.embedder import embed
from app.search.indexer import build_index, load_index, tokenize


def _fixture_laws() -> list[LawChunk]:
    mk = lambda i, art, text: LawChunk.validated(
        id=f"fx-{i}",
        law_name="[TEST-FIXTURE] 테스트법",
        article=art,
        text=text,
        source="[TEST-FIXTURE] pytest",
        source_url="about:test",
    )
    return [
        mk(1, "제1조(설명의무)", "금융상품판매업자는 일반금융소비자에게 상품의 중요한 사항을 설명하여야 한다."),
        mk(2, "제2조(청약철회)", "일반금융소비자는 보장성 상품 계약의 청약을 철회할 수 있다."),
        mk(3, "제3조(광고규제)", "금융상품 광고에는 원금 손실 가능성을 표시하여야 한다."),
    ]


def test_tokenize_korean():
    tokens = tokenize("금융상품판매업자는 설명의무를 진다")
    assert any("설명" in t for t in tokens)


@pytest.mark.slow
def test_index_roundtrip(tmp_path):
    records = _fixture_laws()
    build_index("laws", records, lambda r: r.text, index_dir=tmp_path)
    idx = load_index("laws", index_dir=tmp_path)

    qv = embed(["상품 설명 의무는 어떻게 되나요"])[0]
    vec_hits = idx.search_vec(qv, k=3)
    assert vec_hits[0][0] == "fx-1"

    bm_hits = idx.search_bm25(tokenize("청약 철회"), k=3)
    assert bm_hits[0][0] == "fx-2"
