"""런타임 그래프 싱글턴 + 큐레이션 동기화.

그래프는 부가 레이어다 — 파일이 없으면 None을 반환하고 서비스는 그대로 돈다.

큐레이션(비활성 엣지)을 여기서 읽는 이유:
운영 구성은 고객용(app)과 관리용(app-admin)이 **별도 프로세스**다. 담당자가
관리 인스턴스에서 엣지를 꺼도 고객 인스턴스의 메모리에는 전달되지 않는다.
두 인스턴스가 공유하는 것은 데이터 볼륨의 SQLite뿐이므로, 큐레이션 상태는
공유 DB에서 읽어야 실제로 반영된다. 요청마다 읽지 않도록 TTL 캐시를 둔다.
"""
import sqlite3
import threading
import time

from app.config import settings
from app.graph.context_graph import ContextGraph

_graph: ContextGraph | None = None
_loaded = False

_curation: set[str] = set()
_curation_at: float = 0.0
_curation_lock = threading.Lock()
_curation_failed = False


def get_graph() -> ContextGraph | None:
    global _graph, _loaded
    if not _loaded:
        _loaded = True
        path = settings.INDEX_DIR / "graph.json"
        if path.exists():
            _graph = ContextGraph.load(path)
    return _graph


def _read_curation() -> set[str]:
    """공유 DB에서 비활성 엣지 목록을 읽는다 (읽기 전용, 스키마 생성 안 함).

    DB가 아직 없거나 테이블이 없으면 빈 집합 — 큐레이션이 없다는 뜻이지
    오류가 아니다. 그래프 조회가 DB 상태 때문에 실패하면 안 된다.
    """
    if not settings.DB_PATH.exists():
        return set()
    global _curation_failed
    try:
        conn = sqlite3.connect(f"file:{settings.DB_PATH}?mode=ro", uri=True, timeout=2.0)
        try:
            rows = conn.execute(
                "SELECT edge_id FROM graph_curation WHERE disabled=1"
            ).fetchall()
            _curation_failed = False
            return {r[0] for r in rows}
        finally:
            conn.close()
    except sqlite3.Error as e:
        # 조용히 넘어가면 담당자가 끈 근거가 되살아나 답변에 다시 쓰인다.
        # 큐레이션이 답변에 반영된다는 약속이 무음으로 깨지는 지점이므로 알린다.
        _curation_failed = True
        print(f"[graph] 큐레이션 읽기 실패 — 비활성 엣지가 답변 근거에 "
              f"다시 사용됩니다: {type(e).__name__}: {e}", flush=True)
        return set()


def curation_degraded() -> bool:
    """마지막 큐레이션 조회가 실패했는가 — /api/health가 운영자에게 노출한다."""
    return _curation_failed


def curated_graph(conn=None, force: bool = False) -> ContextGraph | None:
    """큐레이션이 반영된 그래프. 소비자·관리자 경로 공통 진입점.

    conn을 주면 그 연결에서 바로 읽는다 — 관리 라우트는 자기가 쓴 DB를
    그대로 읽어야 토글이 즉시 보인다. TTL 캐시를 태우면 자기가 방금 쓴 값을
    못 보는 창이 생기고, 테스트 DB와 운영 DB가 갈리는 사고도 난다.

    conn 없이 부르면(소비자 경로) 공유 DB를 TTL 캐시로 읽는다.
    """
    graph = get_graph()
    if graph is None:
        return None
    if conn is not None:
        rows = conn.execute(
            "SELECT edge_id FROM graph_curation WHERE disabled=1"
        ).fetchall()
        graph.disabled = {r[0] for r in rows}
        return graph
    global _curation, _curation_at
    now = time.monotonic()
    with _curation_lock:
        # >= 이어야 TTL=0이 "항상 갱신"이 된다. Windows monotonic 해상도가
        # ~15.6ms라 >를 쓰면 같은 틱 안의 재조회가 조용히 캐시를 재사용한다.
        if force or (now - _curation_at) >= settings.GRAPH_CURATION_TTL:
            _curation = _read_curation()
            _curation_at = now
        graph.disabled = _curation
    return graph


def reload_graph() -> ContextGraph | None:
    global _loaded, _graph, _curation_at
    _loaded = False
    _graph = None
    _curation_at = 0.0
    return get_graph()
