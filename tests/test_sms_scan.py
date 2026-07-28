import pytest

from app.agent import redflag
from app.agent.redflag import build_summary, scan_sms_batch, split_messages
from app.corpus.schema import LawChunk
from app.search.retriever import Hit

SMS = (
    "[Web발신] 금융감독원 수사관입니다. 안전계좌로 이체하세요\n\n"
    "(광고) 오늘만 특가 세일 안내 무료수신거부\n\n"
    "엄마 나 폰 고장났어 이 앱 설치해줘 http://x.co"
)


class StubLLM:
    def generate(self, *a, **k):
        return '{"level": "red", "reason": "기관 사칭"}'


@pytest.fixture(autouse=True)
def _stub_law(monkeypatch):
    rec = LawChunk.validated(
        id="law-x", law_name="[TEST-FIXTURE] 법", article="제1조(테스트)", text="본문",
        source="[TEST-FIXTURE] pytest", source_url="about:test",
    )
    monkeypatch.setattr(redflag, "retrieve", lambda kind, q, **kw: [Hit("law-x", 1.0, rec)])


def test_split_messages():
    assert len(split_messages(SMS)) == 3


def test_sms_batch_report():
    report = scan_sms_batch(SMS, StubLLM())
    assert report.input_kind == "sms_batch"
    assert report.summary.total == 3
    assert report.summary.red >= 1
    assert any("118" in a for a in report.summary.actions)
    assert report.summary.top_risk


def test_benign_messages_no_llm_call():
    calls = []

    class Counting:
        def generate(self, *a, **k):
            calls.append(1)
            return '{"level": "red", "reason": "r"}'

    report = scan_sms_batch("점심 뭐 먹을래?\n\n회의 3시로 변경됐어요", Counting())
    assert calls == []
    assert report.summary.red == 0
    assert report.summary.actions == []


def test_document_scan_gets_summary():
    report = redflag.scan("1. 원금 100% 보장!\n2. 영업시간 안내입니다", StubLLM())
    assert report.input_kind == "document"
    assert report.summary.total == 2
    assert "118" not in " ".join(report.summary.actions)  # 문서 스캔엔 신고 문구 없음
