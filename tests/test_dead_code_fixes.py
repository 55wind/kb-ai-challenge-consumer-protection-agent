"""감사에서 찾은 '계산은 하는데 안 쓰이던' 것들의 회귀 방지.

만들어놓고 화면에 안 그리거나, 이름과 다르게 동작하거나, 조용히 죽던 것들이다.
"""
from app.agent import draft as draft_mod
from app.agent.answerer import find_glossary_terms
from app.agent.slots import SLOT_LABEL, SLOT_SCHEMA


class _LLM:
    model = "stub"

    def generate(self, *a, **k):
        return "정리된 문장"


def test_slot_labels_cover_every_slot():
    """화면이 '남은 항목: 누가 · 언제'로 보여주려면 전 슬롯에 이름이 있어야 한다."""
    assert set(SLOT_LABEL) == set(SLOT_SCHEMA), "슬롯과 표시 이름이 어긋난다"
    assert all(v and not v.isascii() for v in SLOT_LABEL.values())


def test_draft_reports_actually_missing_fields():
    """'빠진 항목'이 이름대로 동작해야 한다.

    이전에는 `evidence if missing else evidence`로 삼항 양쪽이 같아,
    무엇이 비었든 항상 전체 증빙 목록만 돌려줬다.
    """
    from app.agent.slots import Slots

    full = {k: "값" for k in SLOT_SCHEMA}
    d_full = draft_mod.build_draft(Slots(filled=full, missing=[]), _LLM(),
                                   category="불완전판매")
    assert d_full.missing_fields == [], "다 채웠는데 빠진 항목이 있다고 한다"
    assert d_full.evidence_checklist, "증빙 체크리스트는 항상 있어야 한다"

    partial = {"who": "본인", "product": "펀드"}
    d_part = draft_mod.build_draft(
        Slots(filled=partial, missing=[k for k in SLOT_SCHEMA if k not in partial]),
        _LLM(), category="불완전판매")
    assert set(d_part.missing_fields) == {SLOT_LABEL[k] for k in SLOT_SCHEMA
                                         if k not in partial}
    # 두 필드가 서로 다른 것을 가리켜야 한다 — 예전엔 같은 값이었다
    assert d_part.missing_fields != d_part.evidence_checklist


def test_glossary_skips_everyday_words():
    """일상어와 겹치는 용어에 풀이가 붙으면 설명이 방해가 된다.

    '불공정 조항 검사'의 '검사'가 금융용어로 잡혀 엉뚱한 풀이가 달렸다.
    """
    for s in ("불공정 조항 검사를 해보세요", "금융감독원 검사 결과", "상품 모집 과정에서"):
        assert find_glossary_terms(s) == [], f"일상어에 풀이가 붙었다: {s}"


def test_glossary_still_explains_real_jargon():
    """오탐을 막느라 진짜 어려운 말까지 죽이면 안 된다."""
    hits = {g.term for g in find_glossary_terms(
        "금융투자상품과 금전신탁의 공시이율을 확인하세요")}
    assert {"금융투자상품", "금전신탁", "공시이율"} <= hits


def test_health_reports_corpus_and_curation(tmp_path, monkeypatch):
    """코퍼스 로딩 실패는 첫 사용자 요청이 아니라 헬스체크가 먼저 잡아야 한다.

    지연 로딩이라 그냥 두면 배포 직후가 아니라 첫 질문에서 500이 난다.
    """
    from fastapi.testclient import TestClient

    from app.server import db
    from app.server.api import AppContext, create_app

    monkeypatch.setenv("ADMIN_ENABLED", "false")
    ctx = AppContext(llm=_LLM(), conn=db.connect(tmp_path / "h.db"))
    body = TestClient(create_app(ctx)).get("/api/health").json()

    assert "corpus" in body["checks"], "코퍼스 상태가 헬스체크에 없다"
    assert "graph" in body
    # 큐레이션이 무음으로 깨졌는지 운영자가 볼 수 있어야 한다
    assert "curation_degraded" in body["graph"]


def test_curation_failure_is_visible(tmp_path, monkeypatch, capsys):
    """큐레이션 조회가 실패하면 조용히 넘어가지 않는다.

    조용히 빈 집합을 돌려주면 담당자가 끈 근거가 답변에 되살아난다.
    운영자가 알 방법이 없으므로 로그와 헬스체크 양쪽으로 드러내야 한다.
    """
    import sqlite3

    from app.config import settings
    from app.graph import runtime
    from app.server import db

    # 실제 DB를 만들어 두고(파일 존재 검사 통과) 연결만 실패시킨다 — DB 잠금 상황
    db_path = tmp_path / "locked.db"
    db.connect(db_path).close()
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(runtime, "_curation_failed", False)
    monkeypatch.setattr(
        runtime.sqlite3, "connect",
        lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    assert runtime._read_curation() == set()
    assert runtime.curation_degraded() is True, "큐레이션 실패가 드러나지 않는다"
    assert "큐레이션" in capsys.readouterr().out, "운영자에게 알리는 로그가 없다"
