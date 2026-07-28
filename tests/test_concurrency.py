import threading

import pytest

from app.config import settings
from app.server import db
from app.server.api import AppContext


class StubLLM:
    model = "stub"

    def generate(self, *a, **k):
        return ""


def _ctx(tmp_path):
    return AppContext(llm=StubLLM(), conn=db.connect(tmp_path / "c.db"))


def test_session_ttl_eviction(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.state("a")
    ctx.last_seen["a"] -= settings.SESSION_TTL_SECONDS + 1
    ctx.state("b")
    assert "a" not in ctx.sessions and "b" in ctx.sessions


def test_session_cap_lru(tmp_path):
    ctx = _ctx(tmp_path)
    for i in range(settings.SESSION_MAX + 5):
        ctx.state(f"s{i}")
    assert len(ctx.sessions) <= settings.SESSION_MAX
    assert f"s{settings.SESSION_MAX + 4}" in ctx.sessions  # 최신 세션은 유지


def test_semaphore_bound():
    ctx = AppContext(llm=StubLLM(), conn=None.__class__)
    for _ in range(settings.LLM_MAX_CONCURRENT):
        assert ctx.llm_slots.acquire(timeout=0.1)
    assert not ctx.llm_slots.acquire(timeout=0.1)  # 슬롯 소진 → 대기/거절 경로
    ctx.llm_slots.release()
    assert ctx.llm_slots.acquire(timeout=0.1)


def test_sqlite_concurrent_writes(tmp_path):
    conn = db.connect(tmp_path / "w.db")
    errors = []

    def write(i):
        try:
            db.add_message(conn, f"s{i}", "user", "x")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(db.list_messages(conn)) == 20
