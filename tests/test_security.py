import pytest
from fastapi.testclient import TestClient

from app.agent.orchestrator import AgentEvent
from app.server import db, security
from app.server.api import AppContext, create_app
from app.server.security import RateLimiter, mask_pii, reset_admin_token_for_tests

TEST_TOKEN = "test-admin-token"


class StubLLM:
    model = "stub"

    def generate(self, *a, **k):
        return ""


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", TEST_TOKEN)
    reset_admin_token_for_tests()

    def fake_handle(state, text, llm, mode="chat"):
        yield AgentEvent("answer", {"text": "답변[1]"})

    monkeypatch.setattr("app.server.api.handle_message", fake_handle)
    ctx = AppContext(llm=StubLLM(), conn=db.connect(tmp_path / "s.db"))
    yield TestClient(create_app(ctx)), ctx
    reset_admin_token_for_tests()


def test_mask_pii():
    assert mask_pii("주민번호 900101-1234567 연락처 010-1234-5678") == \
        "주민번호 ******-******* 연락처 010-****-5678"


def test_mask_account_number():
    assert mask_pii("계좌 12345678901234로 이체") == "계좌 1234**********로 이체"


def test_mask_hyphenated_account_number():
    """하이픈 구분 계좌번호도 마스킹된다 (연속 숫자 정규식만으로는 통과했던 형태)."""
    masked = mask_pii("계좌 123-456-789012 로 보냈어요")
    assert "123-456-789012" not in masked
    assert masked.startswith("계좌 123-4")


def test_mask_pii_keeps_dates_readable():
    """날짜(8자리)는 계좌로 오인해 가리지 않는다 — 담당자 판독성 유지."""
    assert "2026-07-27" in mask_pii("2026-07-27에 가입했습니다")


def test_unknown_session_rejected(client):
    c, _ = client
    assert c.post("/api/chat", json={"session_id": "forged", "text": "hi"}).status_code == 403


def test_session_issue_and_use(client):
    c, _ = client
    sid = c.post("/api/session").json()["session_id"]
    assert c.post("/api/chat", json={"session_id": sid, "text": "hi"}).status_code == 200


def test_admin_requires_token(client):
    c, _ = client
    assert c.get("/api/admin/tickets").status_code == 401
    assert c.get("/api/admin/tickets", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert c.get("/api/admin/tickets", headers={"X-Admin-Token": TEST_TOKEN}).status_code == 200


def test_input_cap(client):
    c, _ = client
    sid = c.post("/api/session").json()["session_id"]
    resp = c.post("/api/chat", json={"session_id": sid, "text": "가" * 20001})
    assert resp.status_code == 422


def test_rate_limiter_unit():
    rl = RateLimiter(limit=3, window=60)
    assert [rl.allow("ip") for _ in range(4)] == [True, True, True, False]
    assert rl.allow("other-ip")  # 키 분리


def test_rate_limit_endpoint(client):
    c, ctx = client
    ctx.rate_limiter.limit = 2
    assert c.post("/api/session").status_code == 200
    assert c.post("/api/session").status_code == 200
    assert c.post("/api/session").status_code == 429


def test_security_headers(client):
    c, _ = client
    r = c.get("/api/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"


def test_admin_disabled_instance(tmp_path, monkeypatch):
    """고객용 인스턴스(ADMIN_ENABLED=false)에는 관리 라우트가 등록되지 않는다."""
    from app.config import settings as cfg

    monkeypatch.setenv("ADMIN_TOKEN", TEST_TOKEN)
    reset_admin_token_for_tests()
    monkeypatch.setattr(cfg, "ADMIN_ENABLED", False)
    ctx = AppContext(llm=StubLLM(), conn=db.connect(tmp_path / "d.db"))
    c = TestClient(create_app(ctx))
    assert c.get("/api/admin/tickets", headers={"X-Admin-Token": TEST_TOKEN}).status_code == 404
    assert c.get("/api/admin/audit", headers={"X-Admin-Token": TEST_TOKEN}).status_code == 404
    assert c.get("/api/health").status_code == 200  # 고객 기능은 정상
    reset_admin_token_for_tests()


def test_admin_ip_allowlist(tmp_path, monkeypatch):
    from app.config import settings as cfg

    monkeypatch.setenv("ADMIN_TOKEN", TEST_TOKEN)
    reset_admin_token_for_tests()
    ctx = AppContext(llm=StubLLM(), conn=db.connect(tmp_path / "ip.db"))
    c = TestClient(create_app(ctx))
    headers = {"X-Admin-Token": TEST_TOKEN}
    # TestClient의 client host는 'testclient' — 허용목록 리터럴 매치로 검증
    monkeypatch.setattr(cfg, "ADMIN_IP_ALLOWLIST", "10.0.0.0/8")
    assert c.get("/api/admin/tickets", headers=headers).status_code == 403
    monkeypatch.setattr(cfg, "ADMIN_IP_ALLOWLIST", "10.0.0.0/8,testclient")
    assert c.get("/api/admin/tickets", headers=headers).status_code == 200
    reset_admin_token_for_tests()


def test_admin_audit_records_actor(tmp_path, monkeypatch):
    """토큰별 신원으로 모든 관리 행위가 감사 기록에 남는다."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("ADMIN_TOKENS", "alice:secret-a,bob:secret-b")
    reset_admin_token_for_tests()
    ctx = AppContext(llm=StubLLM(), conn=db.connect(tmp_path / "a.db"))
    c = TestClient(create_app(ctx))
    assert c.get("/api/admin/tickets", headers={"X-Admin-Token": "secret-a"}).status_code == 200
    assert c.get("/api/admin/stats", headers={"X-Admin-Token": "secret-b"}).status_code == 200
    audit = c.get("/api/admin/audit", headers={"X-Admin-Token": "secret-a"}).json()["audit"]
    actors = [a["actor"] for a in audit]
    assert "alice" in actors and "bob" in actors
    assert any(a["action"] == "GET /api/admin/tickets" for a in audit)
    assert c.get("/api/admin/tickets", headers={"X-Admin-Token": "wrong"}).status_code == 401
    reset_admin_token_for_tests()


def test_pii_masked_in_stored_logs(client):
    c, ctx = client
    sid = c.post("/api/session").json()["session_id"]
    c.post("/api/chat", json={"session_id": sid, "text": "제 번호는 010-9876-5432입니다"})
    msgs = db.list_messages(ctx.conn, sid)
    assert "010-9876-5432" not in msgs[0]["content"]
    assert "010-****-5432" in msgs[0]["content"]


def test_ticket_summary_masks_pii(tmp_path, monkeypatch):
    """담당자 콘솔 티켓 요약에도 원문 PII가 남지 않는다 (C-2)."""
    monkeypatch.setenv("ADMIN_TOKEN", TEST_TOKEN)
    reset_admin_token_for_tests()

    def fake_handle(state, text, llm, mode="chat"):
        yield AgentEvent("classification", {"category": "불완전판매", "is_dispute": True})

    monkeypatch.setattr("app.server.api.handle_message", fake_handle)
    ctx = AppContext(llm=StubLLM(), conn=db.connect(tmp_path / "t.db"))
    c = TestClient(create_app(ctx))
    sid = c.post("/api/session").json()["session_id"]
    c.post("/api/chat", json={
        "session_id": sid,
        "text": "홍길동 010-1234-5678 주민 900101-1234567 계좌 123-456-789012 펀드 손실",
    })
    summaries = " ".join(t["summary"] for t in db.list_tickets(ctx.conn))
    assert "010-1234-5678" not in summaries
    assert "900101-1234567" not in summaries
    assert "123-456-789012" not in summaries
    reset_admin_token_for_tests()


def test_feedback_rejects_unknown_message(client):
    """존재하지 않는 message_id로는 피드백 통계를 오염시킬 수 없다 (M-1)."""
    c, _ = client
    assert c.post("/api/feedback", json={"session_id": "x", "message_id": 999999, "vote": "up"}).status_code == 404


def test_feedback_accepts_real_message(client):
    c, ctx = client
    sid = c.post("/api/session").json()["session_id"]
    c.post("/api/chat", json={"session_id": sid, "text": "질문"})
    mid = db.list_messages(ctx.conn, sid)[-1]["id"]
    assert c.post("/api/feedback", json={"session_id": sid, "message_id": mid, "vote": "up"}).status_code == 200


def test_feedback_rejects_other_session(client):
    """남의 세션 답변에는 투표할 수 없다 — message_id를 훑어도 소용없어야 한다."""
    c, ctx = client
    sid = c.post("/api/session").json()["session_id"]
    c.post("/api/chat", json={"session_id": sid, "text": "질문"})
    mid = db.list_messages(ctx.conn, sid)[-1]["id"]
    other = c.post("/api/session").json()["session_id"]
    r = c.post("/api/feedback", json={"session_id": other, "message_id": mid, "vote": "up"})
    assert r.status_code == 403


def test_feedback_rejects_user_message(client):
    """사용자가 쓴 메시지에는 평가가 붙지 않는다."""
    c, ctx = client
    sid = c.post("/api/session").json()["session_id"]
    c.post("/api/chat", json={"session_id": sid, "text": "질문"})
    user_mid = db.list_messages(ctx.conn, sid)[0]["id"]
    r = c.post("/api/feedback", json={"session_id": sid, "message_id": user_mid, "vote": "up"})
    assert r.status_code == 400


def test_feedback_is_idempotent(client):
    """같은 답변을 여러 번 눌러도 통계가 부풀지 않는다 (마지막 평가만 남는다)."""
    c, ctx = client
    sid = c.post("/api/session").json()["session_id"]
    c.post("/api/chat", json={"session_id": sid, "text": "질문"})
    mid = db.list_messages(ctx.conn, sid)[-1]["id"]
    for vote in ("up", "up", "down"):
        c.post("/api/feedback", json={"session_id": sid, "message_id": mid, "vote": vote})
    assert db.feedback_stats(ctx.conn) == {"down": 1}


def test_feedback_survives_session_eviction(client):
    """세션 목록에서 밀려나도 정상 투표가 403이 되지 않는다.

    발급 목록은 인메모리라 상한 초과·재기동으로 사라진다. 소유 증명은
    '그 답변을 만든 세션 토큰을 아는가'로 충분하다.
    """
    c, ctx = client
    sid = c.post("/api/session").json()["session_id"]
    c.post("/api/chat", json={"session_id": sid, "text": "질문"})
    mid = db.list_messages(ctx.conn, sid)[-1]["id"]
    ctx.issued_sessions.clear()          # 퇴출·재기동 재현
    r = c.post("/api/feedback", json={"session_id": sid, "message_id": mid, "vote": "up"})
    assert r.status_code == 200


def test_feedback_is_rate_limited(client):
    c, ctx = client
    sid = c.post("/api/session").json()["session_id"]
    c.post("/api/chat", json={"session_id": sid, "text": "질문"})
    mid = db.list_messages(ctx.conn, sid)[-1]["id"]
    ctx.rate_limiter.limit = 2
    ctx.rate_limiter._hits.clear()
    codes = [c.post("/api/feedback", json={"session_id": sid, "message_id": mid, "vote": "up"}).status_code
             for _ in range(3)]
    assert codes[-1] == 429


# ── 리버스 프록시 뒤 클라이언트 IP (H-5) ────────────────────────────
def test_forwarded_for_ignored_without_trusted_proxy(monkeypatch):
    """신뢰 프록시 미설정 시 X-Forwarded-For는 무시 — 헤더 위조로 우회 불가."""
    from app.config import settings as cfg
    from app.server.security import client_ip

    monkeypatch.setattr(cfg, "TRUSTED_PROXY_IPS", "")
    req = _FakeRequest("10.1.1.9", {"x-forwarded-for": "203.0.113.7"})
    assert client_ip(req) == "10.1.1.9"


def test_forwarded_for_used_behind_trusted_proxy(monkeypatch):
    from app.config import settings as cfg
    from app.server.security import client_ip

    monkeypatch.setattr(cfg, "TRUSTED_PROXY_IPS", "10.1.1.0/24")
    req = _FakeRequest("10.1.1.9", {"x-forwarded-for": "203.0.113.7, 10.1.1.9"})
    assert client_ip(req) == "203.0.113.7"


def test_rate_limit_separates_clients_behind_proxy(monkeypatch):
    """프록시 뒤에서도 사용자별로 레이트리밋이 분리된다 (전원 429 방지)."""
    from app.config import settings as cfg
    from app.server.security import RateLimiter as RL, client_ip

    monkeypatch.setattr(cfg, "TRUSTED_PROXY_IPS", "10.1.1.0/24")
    rl = RL(limit=1, window=60)
    a = _FakeRequest("10.1.1.9", {"x-forwarded-for": "203.0.113.7"})
    b = _FakeRequest("10.1.1.9", {"x-forwarded-for": "198.51.100.4"})
    assert rl.allow(client_ip(a)) is True
    assert rl.allow(client_ip(b)) is True   # 다른 사용자는 영향받지 않는다
    assert rl.allow(client_ip(a)) is False


class _FakeRequest:
    def __init__(self, host: str, headers: dict[str, str]):
        self.client = type("C", (), {"host": host})()
        self.headers = headers
