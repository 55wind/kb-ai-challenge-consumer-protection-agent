import pytest
from fastapi.testclient import TestClient

from app.agent import orchestrator as orch_mod
from app.agent.orchestrator import AgentEvent, SessionState
from app.server import db
from app.server.api import AppContext, create_app


class StubLLM:
    model = "stub"

    def generate(self, *a, **k):
        return ""


TEST_TOKEN = "api-test-token"


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.server.security import reset_admin_token_for_tests

    monkeypatch.setenv("ADMIN_TOKEN", TEST_TOKEN)
    reset_admin_token_for_tests()
    conn = db.connect(tmp_path / "test.db")

    def fake_handle(state, text, llm, mode="chat"):
        yield AgentEvent("classification", {"category": "예금·계좌", "is_dispute": False})
        yield AgentEvent("answer", {"text": "테스트 답변[1]"})

    monkeypatch.setattr("app.server.api.handle_message", fake_handle)
    ctx = AppContext(llm=StubLLM(), conn=conn)
    c = TestClient(create_app(ctx))
    yield c, ctx
    reset_admin_token_for_tests()


def _sid(c) -> str:
    return c.post("/api/session").json()["session_id"]

def _admin(path, c):
    return c.get(path, headers={"X-Admin-Token": TEST_TOKEN})


def test_health_reports_component_checks(client):
    c, _ = client
    body = c.get("/api/health").json()
    # ok는 실제 의존성(Ollama 등) 상태에 따라 달라짐 — 구조만 검증
    assert set(body["checks"]) >= {"ollama", "indexes", "graph", "frontend"}
    assert "llm_slots_free" in body


def test_chat_sse_order_and_persistence(client):
    c, ctx = client
    sid = _sid(c)
    resp = c.post("/api/chat", json={"session_id": sid, "text": "수수료 문의"})
    assert resp.status_code == 200
    body = resp.text
    assert body.index("disclaimer") < body.index("classification") < body.index("answer")
    assert body.rstrip().endswith("{}")  # done 이벤트
    msgs = db.list_messages(ctx.conn, sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    tickets = db.list_tickets(ctx.conn)
    assert tickets[0]["category"] == "예금·계좌"


def test_feedback_and_stats(client):
    c, ctx = client
    sid = _sid(c)
    c.post("/api/chat", json={"session_id": sid, "text": "질문"})
    mid = db.list_messages(ctx.conn, sid)[-1]["id"]
    assert c.post("/api/feedback", json={"session_id": sid, "message_id": mid, "vote": "up"}).json()["ok"]
    assert c.post("/api/feedback", json={"session_id": sid, "message_id": mid, "vote": "bad"}).status_code == 400
    stats = _admin("/api/admin/stats", c).json()
    assert stats["feedback"]["up"] == 1


def test_draft_requires_slots(client):
    c, ctx = client
    sid = _sid(c)
    assert c.post(f"/api/draft/{sid}", json={}).status_code == 400
    sid2 = _sid(c)
    ctx.state(sid2).slots_filled = {
        "who": "본인", "when": "2025.3", "product": "펀드",
        "incident": "설명 없이 가입", "demand": "배상",
    }
    ctx.state(sid2).category = "불완전판매"
    resp = c.post(f"/api/draft/{sid2}", json={"respondent": "○○은행"})
    assert resp.status_code == 200
    assert "○○은행" in resp.json()["markdown"]


def test_admin_logs(client):
    c, _ = client
    sid = _sid(c)
    c.post("/api/chat", json={"session_id": sid, "text": "로그 확인"})
    logs = _admin(f"/api/admin/logs?session_id={sid}", c).json()
    assert len(logs["messages"]) == 2


# ── 스트림 실패가 무음이 되지 않는다 (H-6) ──────────────────────────
def _stream_events(tmp_path, monkeypatch, failure: Exception):
    from app.server.security import reset_admin_token_for_tests

    reset_admin_token_for_tests()

    def failing_handle(state, text, llm, mode="chat"):
        yield AgentEvent("status", {"message": "찾는 중"})
        raise failure

    monkeypatch.setattr("app.server.api.handle_message", failing_handle)
    ctx = AppContext(llm=StubLLM(), conn=db.connect(tmp_path / "f.db"))
    c = TestClient(create_app(ctx))
    sid = c.post("/api/session").json()["session_id"]
    body = c.post("/api/chat", json={"session_id": sid, "text": "질문"}).text
    reset_admin_token_for_tests()
    return body, ctx


def test_llm_timeout_emits_error_and_done(tmp_path, monkeypatch):
    """CPU 추론 지연으로 LLM이 죽어도 화면이 '생각 중'에 멈추지 않는다."""
    from app.llm.ollama_client import LLMUnavailable

    body, _ = _stream_events(tmp_path, monkeypatch,
                             LLMUnavailable("답변 생성이 제한 시간을 넘겨 중단했습니다"))
    assert "event: error" in body
    assert "제한 시간" in body
    assert body.rstrip().endswith("{}")   # done 이벤트로 정상 종료


def test_unexpected_error_emits_error_and_done(tmp_path, monkeypatch):
    body, _ = _stream_events(tmp_path, monkeypatch, RuntimeError("예기치 못한 실패"))
    assert "event: error" in body
    assert body.rstrip().endswith("{}")


def test_llm_slot_released_after_failure(tmp_path, monkeypatch):
    """실패해도 세마포어가 반환되어 다음 사용자가 막히지 않는다."""
    from app.llm.ollama_client import LLMUnavailable

    _, ctx = _stream_events(tmp_path, monkeypatch, LLMUnavailable("timeout"))
    from app.config import settings as cfg

    assert ctx.llm_slots._value == cfg.LLM_MAX_CONCURRENT


def test_startup_warms_embedder(monkeypatch, tmp_path):
    """임베딩 모델은 기동 때 올린다 — 실패가 첫 사용자 질문에서 터지지 않게.

    지연 로드였을 때 메모리 부족이 나면 프로세스가 통째로 죽고, 브라우저에는
    이유 없는 "network error"만 남았다.
    """
    from app.server import api as api_mod

    called = []
    monkeypatch.setattr(api_mod, "_warm_embedder", lambda: called.append(True))
    monkeypatch.setattr(api_mod.settings, "DB_PATH", tmp_path / "warm.db")
    api_mod.create_app()
    assert called, "기동에서 임베딩 워밍업이 호출되지 않았다"


def test_startup_survives_embedder_failure(monkeypatch, capsys):
    """모델을 못 올려도 기동은 계속된다 — 운영자가 /api/health로 원인을 봐야 한다."""
    from app.search import embedder
    from app.server import api as api_mod

    def boom(_texts):
        raise OSError("이 작업을 완료하기 위한 페이징 파일이 너무 작습니다.")

    monkeypatch.setattr(embedder, "embed", boom)
    api_mod._warm_embedder()          # 예외가 밖으로 나오면 기동이 죽는다
    out = capsys.readouterr().out
    assert "임베딩 모델 적재 실패" in out
    assert "메모리" in out, "운영자가 다음에 뭘 할지 알 수 있어야 한다"
