import pytest

from app.agent import answerer
from app.agent.answerer import Answer, UngroundedAnswer, answer, answer_with_retry
from app.corpus.schema import LawChunk
from app.search.retriever import Hit


class StubLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system, prompt, json_mode=False, timeout=120):
        self.calls += 1
        return self.responses.pop(0)


def _hit() -> Hit:
    rec = LawChunk.validated(
        id="law-19", law_name="[TEST-FIXTURE] 금소법", article="제19조(설명의무)",
        text="금융상품판매업자등은 중요사항을 설명하여야 한다.",
        source="[TEST-FIXTURE] pytest", source_url="about:test",
    )
    return Hit(record_id="law-19", score=1.0, record=rec)


@pytest.fixture(autouse=True)
def _no_glossary(monkeypatch):
    monkeypatch.setattr(answerer, "_load_glossary", lambda: {"설명의무": "상품의 중요사항을 설명할 의무"})


def test_answer_maps_citations():
    llm = StubLLM(["판매자에게는 설명의무가 있어 중요사항을 설명해야 합니다[1]."])
    result = answer("설명의무가 뭐죠", [_hit()], llm)
    assert isinstance(result, Answer)
    assert result.citations[0].record_id == "law-19"
    assert result.glossary[0].term == "설명의무"


def test_answer_rejects_uncited():
    llm = StubLLM(["설명해야 합니다."])
    with pytest.raises(UngroundedAnswer):
        answer("설명의무가 뭐죠", [_hit()], llm)


def test_answer_ignores_out_of_range_citation():
    llm = StubLLM(["설명해야 합니다[9]."])
    with pytest.raises(UngroundedAnswer):
        answer("설명의무가 뭐죠", [_hit()], llm)


def test_answer_with_retry_recovers():
    llm = StubLLM(["인용 없음", "재시도 답변[1]"])
    result = answer_with_retry("설명의무가 뭐죠", [_hit()], llm)
    assert llm.calls == 2
    assert result.citations
