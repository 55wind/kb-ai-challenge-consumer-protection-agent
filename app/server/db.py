"""SQLite 저장소: 티켓, 메시지, 피드백, 문서 메타. 로컬 파일 전용(외부 전송 없음).

다중 스레드 접근: WAL 저널 + 모듈 수준 쓰기 락으로 경합을 직렬화한다.
"""
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings

_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    vote TEXT NOT NULL CHECK(vote IN ('up', 'down')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_curation (
    edge_id TEXT PRIMARY KEY,
    disabled INTEGER NOT NULL DEFAULT 1,
    reason TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS admin_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL DEFAULT '',
    ip TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or settings.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def add_message(conn, session_id: str, role: str, content: str) -> int:
    with _write_lock:
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, _now()),
        )
        conn.commit()
        return cur.lastrowid


def upsert_ticket(conn, session_id: str, category: str, summary: str) -> int:
    with _write_lock:
        row = conn.execute(
            "SELECT id FROM tickets WHERE session_id=? AND status='open'", (session_id,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE tickets SET category=?, summary=? WHERE id=?",
                (category, summary, row["id"]),
            )
            conn.commit()
            return row["id"]
        cur = conn.execute(
            "INSERT INTO tickets (session_id, category, summary, created_at) VALUES (?,?,?,?)",
            (session_id, category, summary, _now()),
        )
        conn.commit()
        return cur.lastrowid


def list_tickets(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM tickets ORDER BY created_at DESC").fetchall()]


def category_stats(conn) -> dict[str, int]:
    rows = conn.execute(
        "SELECT category, COUNT(*) n FROM tickets GROUP BY category").fetchall()
    return {r["category"]: r["n"] for r in rows}


def get_message_role_session(conn, message_id: int) -> tuple[str, str] | None:
    """피드백 대상 메시지의 (역할, 세션) — 존재하지 않으면 None."""
    row = conn.execute(
        "SELECT role, session_id FROM messages WHERE id=?", (message_id,)
    ).fetchone()
    return (row["role"], row["session_id"]) if row else None


def set_feedback(conn, message_id: int, vote: str) -> None:
    """메시지당 최신 평가 1건만 남긴다.

    누적 INSERT면 같은 사용자가 버튼을 여러 번 눌러 통계를 부풀릴 수 있다.
    스키마 변경(UNIQUE 인덱스)은 기존 DB에 중복이 있으면 생성 자체가 실패하므로,
    쓰기 시점에 이전 평가를 지우는 방식으로 처리한다.
    """
    with _write_lock:
        conn.execute("DELETE FROM feedback WHERE message_id=?", (message_id,))
        conn.execute(
            "INSERT INTO feedback (message_id, vote, created_at) VALUES (?,?,?)",
            (message_id, vote, _now()),
        )
        conn.commit()


def list_messages(conn, session_id: str | None = None, limit: int = 200) -> list[dict]:
    if session_id:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY id LIMIT ?",
            (session_id, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def sync_documents(conn) -> int:
    """코퍼스 JSONL의 문서 메타를 documents 테이블로 동기화."""
    from app.corpus.loader import load_corpus

    n = 0
    for kind in ("laws", "cases", "procedures", "kb_products", "interpretations"):
        try:
            for r in load_corpus(kind):
                # KB 상품 청크는 title/article이 없다 — 상품명·구획으로 제목을 만든다
                product = getattr(r, "product_name", None)
                if product:
                    section = getattr(r, "section", "") or getattr(r, "doc_type", "")
                    title = f"{product} · {section}".strip(" ·")
                else:
                    title = getattr(r, "title", None) or getattr(r, "article", r.id)
                conn.execute(
                    "INSERT OR REPLACE INTO documents (id, kind, title, source, source_url, ingested_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (r.id, kind, title, r.source, r.source_url, _now()),
                )
                n += 1
        except Exception as e:  # noqa: BLE001
            # 조용히 넘기면 관리 화면 지식베이스에서 그 종류가 통째로 사라지는데
            # 원인 표시가 없다. 기동 시 1회만 도는 코드라 발견도 늦는다.
            print(f"[documents] '{kind}' 동기화 실패 — 관리 화면에서 이 종류가 "
                  f"비어 보입니다: {type(e).__name__}: {e}", flush=True)
            continue
    conn.commit()
    return n


def list_documents(conn, kind: str | None = None) -> list[dict]:
    if kind:
        rows = conn.execute("SELECT * FROM documents WHERE kind=? ORDER BY id", (kind,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM documents ORDER BY kind, id").fetchall()
    return [dict(r) for r in rows]


def add_audit(conn, actor: str, action: str, target: str = "", ip: str = "") -> None:
    with _write_lock:
        conn.execute(
            "INSERT INTO admin_audit (actor, action, target, ip, created_at) VALUES (?,?,?,?,?)",
            (actor, action, target[:300], ip, _now()),
        )
        conn.commit()


def list_audit(conn, limit: int = 200) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM admin_audit ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def set_edge_curation(conn, edge_id: str, disabled: bool, reason: str = "") -> None:
    with _write_lock:
        conn.execute(
            "INSERT OR REPLACE INTO graph_curation (edge_id, disabled, reason, updated_at) "
            "VALUES (?,?,?,?)",
            (edge_id, 1 if disabled else 0, reason, _now()),
        )
        conn.commit()


def disabled_edge_ids(conn) -> set[str]:
    rows = conn.execute("SELECT edge_id FROM graph_curation WHERE disabled=1").fetchall()
    return {r["edge_id"] for r in rows}


def feedback_stats(conn) -> dict[str, int]:
    rows = conn.execute("SELECT vote, COUNT(*) n FROM feedback GROUP BY vote").fetchall()
    return {r["vote"]: r["n"] for r in rows}
