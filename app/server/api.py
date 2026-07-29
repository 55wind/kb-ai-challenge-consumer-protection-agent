"""FastAPI 서버: 사용자 챗(SSE) + 스캔 + 초안 + 관리자 API.

동시접속 설계: 파이프라인(동기·블로킹)은 스레드풀에서 실행하고, CPU LLM은
세마포어(LLM_MAX_CONCURRENT)로 보호한다. 세션은 TTL·상한으로 퇴출한다.
"""
import json
import threading
import time
import traceback
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from app.server.security import (
    RateLimiter,
    SecurityHeadersMiddleware,
    check_admin_ip,
    client_ip,
    mask_pii,
    new_session_id,
    resolve_admin_actor,
)

from app.agent.orchestrator import (
    DISCLAIMER,
    SessionState,
    build_draft_for,
    handle_message,
)
from app.config import settings
from app.llm.ollama_client import LLMUnavailable, OllamaClient
from app.server import db


class ChatRequest(BaseModel):
    session_id: str
    text: str = Field(max_length=settings.INPUT_MAX_CHARS)
    mode: str = "chat"


class FeedbackRequest(BaseModel):
    message_id: int
    vote: str
    session_id: str = ""


class DraftRequest(BaseModel):
    respondent: str = Field(default="", max_length=100)


@dataclass
class AppContext:
    llm: OllamaClient = field(default_factory=OllamaClient)
    sessions: dict[str, SessionState] = field(default_factory=dict)
    conn: object = None
    llm_slots: threading.BoundedSemaphore = field(
        default_factory=lambda: threading.BoundedSemaphore(settings.LLM_MAX_CONCURRENT)
    )
    last_seen: dict[str, float] = field(default_factory=dict)
    issued_sessions: set[str] = field(default_factory=set)
    rate_limiter: RateLimiter = field(
        default_factory=lambda: RateLimiter(limit=settings.RATE_LIMIT_PER_MINUTE)
    )

    def state(self, session_id: str) -> SessionState:
        now = time.monotonic()
        # TTL 만료 세션 퇴출
        expired = [
            sid for sid, seen in self.last_seen.items()
            if now - seen > settings.SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self.sessions.pop(sid, None)
            self.last_seen.pop(sid, None)
        # 상한 초과 시 가장 오래된 세션 제거 (LRU)
        while len(self.sessions) >= settings.SESSION_MAX and session_id not in self.sessions:
            oldest = min(self.last_seen, key=self.last_seen.get)
            self.sessions.pop(oldest, None)
            self.last_seen.pop(oldest, None)
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        self.last_seen[session_id] = now
        return self.sessions[session_id]


def _warm_embedder() -> None:
    """임베딩 모델을 기동 때 올려 둔다.

    두 가지를 얻는다.

    1. **실패가 사용자 앞에서 터지지 않는다.** 지연 로드였을 때는 첫 질문이
       들어온 뒤에야 2GB 모델을 올렸다. 메모리가 부족하면 그 순간 프로세스가
       죽고, 브라우저에는 이유 없는 "network error"만 뜬다. 서버 로그에도
       파이썬 예외가 아니라 접근 위반이 남아 원인을 알기 어렵다. 기동 때
       올리면 같은 실패가 배포 시점에, 로그와 함께 드러난다.
    2. 첫 사용자가 콜드 로드 10초대를 혼자 부담하지 않는다.

    실패해도 기동은 계속한다 — 검색이 안 되는 상태라도 /api/health가 살아 있어야
    운영자가 원인을 조회할 수 있다. health의 `indexes` 항목이 이미 이를 보고한다.
    """
    try:
        from app.search.embedder import embed
        embed(["기동 확인"])
    except Exception as e:  # noqa: BLE001 - 기동을 막지 않는다
        print(f"[기동] 임베딩 모델 적재 실패 — 검색이 동작하지 않습니다: "
              f"{type(e).__name__}: {e}", flush=True)
        print("[기동] 메모리 부족(OSError 1455 등)이면 가상 메모리를 늘리거나 "
              "다른 인스턴스를 줄이세요. 모델 적재에 2GB 이상이 필요합니다.", flush=True)


def create_app(context: AppContext | None = None) -> FastAPI:
    app = FastAPI(title="금융 소비자 보호 에이전트")
    ctx = context or AppContext()
    if ctx.conn is None:
        ctx.conn = db.connect()
        db.sync_documents(ctx.conn)
        _warm_embedder()
    app.state.ctx = ctx

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Admin-Token"],
    )
    app.add_middleware(SecurityHeadersMiddleware)

    def _check_rate(request: Request) -> None:
        key = client_ip(request) or "unknown"
        if not ctx.rate_limiter.allow(key):
            raise HTTPException(429, "요청이 너무 잦습니다. 잠시 후 다시 시도해 주세요.")

    def _check_session(session_id: str) -> None:
        if session_id not in ctx.issued_sessions:
            raise HTTPException(403, "유효하지 않은 세션입니다. 페이지를 새로고침해 주세요.")

    @app.post("/api/session")
    def create_session(request: Request):
        _check_rate(request)
        sid = new_session_id()
        ctx.issued_sessions.add(sid)
        # 세션 상한과 동일한 상한으로 발급 목록도 제한
        while len(ctx.issued_sessions) > settings.SESSION_MAX * 2:
            ctx.issued_sessions.pop()
        return {"session_id": sid}

    @app.post("/api/chat")
    async def chat(req: ChatRequest, request: Request):
        _check_rate(request)
        _check_session(req.session_id)
        state = ctx.state(req.session_id)
        db.add_message(ctx.conn, req.session_id, "user", mask_pii(req.text))

        def event_stream():
            yield {"event": "disclaimer", "data": json.dumps({"message": DISCLAIMER}, ensure_ascii=False)}
            acquired = ctx.llm_slots.acquire(timeout=settings.LLM_ACQUIRE_TIMEOUT)
            if not acquired:
                yield {"event": "busy", "data": json.dumps(
                    {"message": "상담 요청이 많아 잠시 대기 중입니다. 다시 시도합니다..."},
                    ensure_ascii=False)}
                acquired = ctx.llm_slots.acquire(timeout=settings.LLM_ACQUIRE_TIMEOUT)
                if not acquired:
                    yield {"event": "error", "data": json.dumps(
                        {"message": "현재 상담 요청이 많습니다. 잠시 후 다시 시도해 주세요."},
                        ensure_ascii=False)}
                    yield {"event": "done", "data": "{}"}
                    return
            try:
                for ev in handle_message(state, req.text, ctx.llm, mode=req.mode):
                    if ev.type == "answer":
                        mid = db.add_message(ctx.conn, req.session_id, "assistant",
                                             mask_pii(ev.payload["text"]))
                        ev.payload["message_id"] = mid
                    if ev.type == "classification":
                        # 담당자 콘솔에 뜨는 티켓 요약도 로그와 동일하게 마스킹 후 저장
                        db.upsert_ticket(ctx.conn, req.session_id, ev.payload["category"],
                                         mask_pii(req.text)[:200])
                    yield {"event": ev.type, "data": json.dumps(ev.payload, ensure_ascii=False)}
            except LLMUnavailable as e:
                # CPU 추론 지연(ReadTimeout) 등 — 무음 종료 대신 사용자에게 사유를 알린다.
                yield {"event": "error", "data": json.dumps({"message": str(e)}, ensure_ascii=False)}
            except Exception:  # noqa: BLE001 - 어떤 실패도 스트림을 조용히 끊지 않는다
                traceback.print_exc()
                yield {"event": "error", "data": json.dumps(
                    {"message": "처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."},
                    ensure_ascii=False)}
            finally:
                ctx.llm_slots.release()
            yield {"event": "done", "data": "{}"}

        # 동기 파이프라인을 워커 스레드에서 돌려 이벤트루프 비차단
        return EventSourceResponse(iterate_in_threadpool(event_stream()))

    @app.post("/api/draft/{session_id}")
    def draft(session_id: str, req: DraftRequest, request: Request):
        _check_rate(request)
        _check_session(session_id)
        state = ctx.state(session_id)
        if not state.slots_filled:
            raise HTTPException(400, "사실관계가 아직 정리되지 않았습니다")
        try:
            ev = build_draft_for(state, ctx.llm, respondent=req.respondent)
        except LLMUnavailable as e:
            raise HTTPException(503, str(e)) from e
        return ev.payload

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest, request: Request):
        _check_rate(request)
        if req.vote not in ("up", "down"):
            raise HTTPException(400, "vote must be up|down")
        # 소유권 검증: 호출자가 그 메시지를 만든 세션을 알고 있어야 한다.
        # 메시지 쪽 세션만 보면 아무나 message_id를 훑어 남의 답변에 투표할 수 있고,
        # 발급 목록(in-memory)으로 검사하면 세션 퇴출·재기동 뒤 정상 투표가 403이 된다.
        # 세션 토큰은 서버 발급 난수라 '알고 있다' 자체가 소유 증명이다.
        owner = db.get_message_role_session(ctx.conn, req.message_id)
        if owner is None:
            raise HTTPException(404, "존재하지 않는 메시지입니다")
        role, session_id = owner
        if role != "assistant":
            raise HTTPException(400, "답변 메시지에만 평가할 수 있습니다")
        if not req.session_id or req.session_id != session_id:
            raise HTTPException(403, "이 답변에 대한 평가 권한이 없습니다")
        db.set_feedback(ctx.conn, req.message_id, req.vote)
        return {"ok": True}

    # ── 관리 기능 (정보보호: 별도 인스턴스 분리 스위치) ──
    # 고객용 인스턴스는 ADMIN_ENABLED=false로 기동 — 관리 라우트가 아예 등록되지 않는다.
    # 관리용 인스턴스는 내부망 전용 + IP 허용목록 + 토큰별 신원 + 전 행위 감사 기록.
    def admin_access(request: Request, x_admin_token: str = Header(default="")) -> str:
        client_host = client_ip(request)
        if not check_admin_ip(client_host):
            raise HTTPException(403, "허용되지 않은 접속 위치입니다")
        actor = resolve_admin_actor(x_admin_token)
        if actor is None:
            raise HTTPException(401, "관리자 토큰이 올바르지 않습니다")
        db.add_audit(ctx.conn, actor, f"{request.method} {request.url.path}",
                     target=str(request.query_params) if request.query_params else "",
                     ip=client_host)
        return actor

    if settings.ADMIN_ENABLED and not settings.ADMIN_IP_ALLOWLIST.strip():
        print("[보안 경고] ADMIN_ENABLED=true인데 ADMIN_IP_ALLOWLIST가 비어 있습니다. "
              "관리 화면이 접속 위치 제한 없이 열립니다 — 내부망 대역을 지정하세요.")

    if settings.ADMIN_ENABLED:
        @app.get("/api/admin/tickets", dependencies=[Depends(admin_access)])
        def tickets():
            return {"tickets": db.list_tickets(ctx.conn)}

        @app.get("/api/admin/stats", dependencies=[Depends(admin_access)])
        def stats():
            return {
                "categories": db.category_stats(ctx.conn),
                "feedback": db.feedback_stats(ctx.conn),
            }

        @app.get("/api/admin/documents", dependencies=[Depends(admin_access)])
        def documents(kind: str | None = None):
            return {"documents": db.list_documents(ctx.conn, kind)}

        @app.get("/api/admin/logs", dependencies=[Depends(admin_access)])
        def logs(session_id: str | None = None):
            return {"messages": db.list_messages(ctx.conn, session_id)}

        @app.get("/api/admin/audit", dependencies=[Depends(admin_access)])
        def audit_log(limit: int = 200):
            return {"audit": db.list_audit(ctx.conn, min(limit, 1000))}

    @app.get("/api/health")
    def health():
        """운영 모니터링용 상세 헬스체크 — 의존 구성요소 상태를 개별 보고."""
        import httpx as _httpx

        checks: dict[str, object] = {}
        try:
            r = _httpx.get(f"{settings.OLLAMA_BASE_URL}/api/version", timeout=2)
            checks["ollama"] = r.status_code == 200
        except Exception:  # noqa: BLE001
            checks["ollama"] = False
        # 인덱스 부재는 이제 NoEvidenceFound로 흡수돼 사용자에게는 "근거 없음"으로만
        # 보인다. 그래서 어떤 인덱스가 빠졌는지는 여기서 개별로 알려야 운영자가 안다.
        # kb_products가 빠지면 KB 약관 인용 경로만 조용히 사라진다 — 필수와 구분한다.
        _required = ("laws", "cases", "procedures")
        _optional = ("kb_products",)
        missing = [k for k in _required + _optional
                   if not (settings.INDEX_DIR / f"{k}.faiss").exists()]
        checks["indexes"] = not [k for k in missing if k in _required]
        checks["indexes_missing"] = missing
        checks["graph"] = (settings.INDEX_DIR / "graph.json").exists()
        checks["frontend"] = (settings.WEB_DIST / "index.html").exists()
        # 코퍼스는 지연 로딩이라, 스키마에 안 맞는 레코드가 섞이면 배포 직후가
        # 아니라 **첫 사용자 요청에서** 500이 난다. 월간 갱신에서 새 결정 라벨이
        # 들어오는 경우가 그렇다. 여기서 먼저 읽어 헬스체크가 롤백 신호가 되게 한다.
        from app.corpus.loader import load_corpus

        corpus_error = ""
        try:
            for _k in ("laws", "cases", "procedures", "kb_products"):
                load_corpus(_k)
        except Exception as e:  # noqa: BLE001
            corpus_error = f"{type(e).__name__}: {e}"
        checks["corpus"] = not corpus_error
        if corpus_error:
            checks["corpus_error"] = corpus_error[:300]
        ok = bool(checks["ollama"] and checks["indexes"] and not missing
                  and checks["corpus"])
        # 그래프가 답변에 실제로 관여하는지, 큐레이션이 몇 건 반영 중인지 —
        # 인스턴스가 분리돼 있어 고객용에서도 확인할 수 있어야 한다.
        from app.graph.runtime import curated_graph as _cg
        from app.graph.runtime import curation_degraded

        g = _cg(ctx.conn)
        graph_info = {
            "expansion": settings.GRAPH_EXPANSION,
            "loaded": g is not None,
            "disabled_edges": len(g.disabled) if g else 0,
            "curation_ttl_s": settings.GRAPH_CURATION_TTL,
            # True면 담당자가 끈 근거가 답변에 되살아나고 있다는 뜻
            "curation_degraded": curation_degraded(),
        }
        return {"ok": ok, "model": ctx.llm.model, "checks": checks,
                "graph": graph_info,
                "llm_slots_free": ctx.llm_slots._value}  # noqa: SLF001

    # ── 컨텍스트 그래프 (Knowledge Explorer) ──
    from dataclasses import asdict

    from app.graph.runtime import curated_graph, reload_graph

    def _graph_or_404():
        # 관리 화면은 자기가 쓴 연결에서 큐레이션을 읽는다 — 방금 토글한 값이
        # 즉시 보여야 한다. 고객 인스턴스는 같은 함수를 conn 없이 불러
        # 공유 DB를 TTL로 따라온다.
        graph = curated_graph(ctx.conn)
        if graph is None:
            raise HTTPException(404, "그래프가 없습니다. scripts/build_graph.py를 먼저 실행하세요.")
        return graph

    if settings.ADMIN_ENABLED:
        @app.get("/api/admin/graph/stats", dependencies=[Depends(admin_access)])
        def graph_stats():
            return _graph_or_404().stats()

        @app.get("/api/admin/graph/search", dependencies=[Depends(admin_access)])
        def graph_search(q: str):
            return {"nodes": [asdict(n) for n in _graph_or_404().search(q)]}

        @app.get("/api/admin/graph/neighbors", dependencies=[Depends(admin_access)])
        def graph_neighbors(id: str):
            graph = _graph_or_404()
            if id not in graph.nodes:
                raise HTTPException(404, "노드가 없습니다")
            # 관리 화면에는 비활성 엣지도 보여야 함 → 필터를 끈 채 조회하고 플래그만 부착.
            # 공유 그래프의 disabled를 건드리지 않는다 (동시 요청이 서로 덮어썼던 경로).
            disabled_ids = db.disabled_edge_ids(ctx.conn)
            items = [
                {"edge": asdict(e), "node": asdict(n), "disabled": e.id in disabled_ids}
                for e, n in graph.neighbors(id, disabled=set())
            ]
            return {"node": asdict(graph.nodes[id]), "neighbors": items}

        @app.get("/api/admin/graph/subgraph", dependencies=[Depends(admin_access)])
        def graph_subgraph(id: str | None = None, hops: int = 1, limit: int = 60,
                           exclude: str = ""):
            """그래프 시각화용 서브그래프.

            id를 주면 그 노드 중심, 없으면 유형(category) 노드들을 씨앗으로 개관을 낸다.
            관리 화면은 큐레이션 대상이므로 비활성 엣지도 함께 보내고 플래그로 구분한다.
            exclude는 화면에서 걷어낼 관계 종류(쉼표 구분) — similar_to처럼 팬아웃이 큰
            관계를 빼야 그래프가 읽힌다.
            """
            graph = _graph_or_404()
            if id is not None and id not in graph.nodes:
                raise HTTPException(404, "노드가 없습니다")
            seeds = [id] if id else [n.id for n in graph.nodes.values() if n.type == "category"]
            skip = {r.strip() for r in exclude.split(",") if r.strip()}
            disabled_ids = db.disabled_edge_ids(ctx.conn)
            sub = graph.subgraph(seeds, hops=max(1, min(hops, 3)),
                                 max_nodes=max(10, min(limit, 300)),
                                 exclude_rels=skip, disabled=set())
            for e in sub["edges"]:
                e["disabled"] = e["id"] in disabled_ids
            sub["focus"] = id
            return sub

        @app.post("/api/admin/graph/edge/{edge_id}/toggle")
        def graph_edge_toggle(edge_id: str, request: Request, body: dict | None = None,
                              actor: str = Depends(admin_access)):
            graph = _graph_or_404()
            if edge_id not in graph.edges:
                raise HTTPException(404, "엣지가 없습니다")
            disabled = edge_id not in db.disabled_edge_ids(ctx.conn)
            reason = (body or {}).get("reason", "")
            db.set_edge_curation(ctx.conn, edge_id, disabled, f"{reason} (by {actor})")
            # 다른 인스턴스(고객용)는 공유 DB를 읽어 GRAPH_CURATION_TTL 안에 따라온다.
            return {"edge_id": edge_id, "disabled": disabled}

        @app.post("/api/admin/graph/rebuild", dependencies=[Depends(admin_access)])
        def graph_rebuild():
            from scripts.build_graph import build

            g = build()
            g.save()
            reload_graph()
            return {"ok": True, **g.stats()}

    # ── 프로덕션 정적 프론트 서빙 (web/dist 존재 시) ──
    # 고객은 단일 URL로 접속: /chat(상담), /admin(담당자) — SPA 라우팅 폴백 포함.
    if settings.WEB_DIST.exists():
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles

        assets = settings.WEB_DIST / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            # 경로 탈출 차단: URL 인코딩된 ..%2f 등은 라우트 파라미터 단계에서 이미
            # 디코딩되지만 정규화되지 않는다. resolve() 후 dist 하위인지 반드시 확인한다.
            root = settings.WEB_DIST.resolve()
            if path:
                try:
                    candidate = (root / path).resolve()
                except (OSError, ValueError):
                    candidate = None
                if (candidate is not None and candidate.is_relative_to(root)
                        and candidate.is_file()):
                    return FileResponse(candidate)
            return FileResponse(root / "index.html")

    return app
