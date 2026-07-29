import { lazy, Suspense, useEffect, useState } from "react";
import { admin } from "../lib/api";
import { IconBot, IconThumbsDown, IconThumbsUp, IconUser } from "../components/icons";
import { NODE_TYPE_LEGEND } from "../components/graphTypes";
import type { GraphEdge, GraphNode } from "../components/graphTypes";

// 시각화 라이브러리(cytoscape)는 그래프 탭을 열 때만 내려받는다 —
// 소비자 화면 번들에 섞이지 않도록 분리
const GraphCanvas = lazy(() => import("../components/GraphCanvas"));

type Tab = "tickets" | "documents" | "logs" | "graph" | "audit";

const REL_KO: Record<string, string> = {
  cites: "인용", categorized_as: "분류", similar_to: "유사", grounded_in: "근거",
  applies_to: "적용", remedied_by: "구제절차", escalates_to: "다음 단계", governed_by: "준거",
};

export default function Admin() {
  const [tab, setTab] = useState<Tab>("tickets");
  const [token, setToken] = useState(() => localStorage.getItem("admin_token") ?? "");
  const [authed, setAuthed] = useState(false);
  const [authError, setAuthError] = useState("");
  const [tickets, setTickets] = useState<any[]>([]);
  const [stats, setStats] = useState<{ categories: Record<string, number>; feedback: Record<string, number> }>({
    categories: {},
    feedback: {},
  });
  const [docs, setDocs] = useState<any[]>([]);
  const [docKind, setDocKind] = useState("");
  const [logs, setLogs] = useState<any[]>([]);
  const [gStats, setGStats] = useState<any>(null);
  const [gQuery, setGQuery] = useState("");
  const [gNodes, setGNodes] = useState<any[]>([]);
  const [gFocus, setGFocus] = useState<any>(null);
  const [gBusy, setGBusy] = useState(false);
  const [auditRows, setAuditRows] = useState<any[]>([]);
  // 그래프 시각화 상태
  const [gGraph, setGGraph] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [gFocusId, setGFocusId] = useState<string | null>(null);
  const [gHops, setGHops] = useState(1);
  const [gEdge, setGEdge] = useState<GraphEdge | null>(null);
  const [gView, setGView] = useState<"graph" | "table">("graph");
  // 집중 보기: 통계·검색 줄을 접고 캔버스에 화면을 몰아준다 (복기용)
  const [gFocusMode, setGFocusMode] = useState(false);
  // 유사(similar_to)는 팬아웃이 커서 기본으로 접는다 — 켜면 전부 보인다
  const [gHidden, setGHidden] = useState<string[]>(["similar_to"]);

  const refreshGraphStats = () =>
    admin.graphStats(token).then(setGStats).catch(() => setGStats(null));

  async function loadSubgraph(id?: string, hops = gHops, hidden = gHidden) {
    const d = await admin.graphSubgraph(token, id, hops, hidden);
    setGGraph({ nodes: d.nodes, edges: d.edges });
    setGFocusId(d.focus ?? null);
    setGEdge(null);
  }

  async function graphSearch() {
    if (!gQuery.trim()) return;
    const d = await admin.graphSearch(token, gQuery.trim());
    setGNodes(d.nodes);
    setGFocus(null);
  }

  async function openNode(id: string) {
    setGFocus(await admin.graphNeighbors(token, id));
    await loadSubgraph(id);
  }

  async function toggleEdge(edgeId: string) {
    const reason = window.prompt("사유 (선택):") ?? "";
    await admin.graphToggleEdge(token, edgeId, reason);
    if (gFocus) await openNode(gFocus.node.id);
    else if (gFocusId) await loadSubgraph(gFocusId);
    refreshGraphStats();
  }

  async function rebuildGraph() {
    setGBusy(true);
    try {
      await admin.graphRebuild(token);
      await refreshGraphStats();
    } finally {
      setGBusy(false);
    }
  }

  useEffect(() => {
    if (!token) return;
    admin
      .tickets(token)
      .then((d) => {
        setTickets(d.tickets);
        setAuthed(true);
        setAuthError("");
        localStorage.setItem("admin_token", token);
        return admin.stats(token);
      })
      .then((s) => s && setStats(s))
      .catch(() => {
        setAuthed(false);
        setAuthError("토큰이 올바르지 않습니다. 서버 콘솔에 출력된 관리자 토큰을 입력하세요.");
      });
  }, [token]);

  useEffect(() => {
    if (!authed) return;
    if (tab === "documents") admin.documents(token, docKind || undefined).then((d) => setDocs(d.documents)).catch(() => {});
    if (tab === "logs") admin.logs(token).then((d) => setLogs(d.messages)).catch(() => {});
    if (tab === "graph") {
      refreshGraphStats();
      // 처음 들어오면 유형 노드 기준 개관을 그린다
      if (!gGraph) loadSubgraph(undefined, 1).catch(() => {});
    }
    if (tab === "audit") admin.audit(token).then((d) => setAuditRows(d.audit)).catch(() => {});
  }, [tab, docKind, authed, token]);

  const totalCat = Object.values(stats.categories).reduce((a, b) => a + b, 0) || 1;

  if (!authed) {
    return (
      <div className="admin-page">
        <h1>KB 소비자보호 담당자 워크벤치</h1>
        <div className="draft-form" style={{ maxWidth: 480 }}>
          <input
            type="password"
            placeholder="관리자 토큰 (서버 콘솔에 출력됨)"
            defaultValue={token}
            onKeyDown={(e) => {
              if (e.key === "Enter") setToken((e.target as HTMLInputElement).value.trim());
            }}
          />
          <button onClick={(e) => {
            const inp = (e.currentTarget.previousSibling as HTMLInputElement);
            setToken(inp.value.trim());
          }}>접속</button>
        </div>
        {authError && <p style={{ marginTop: 10, color: "var(--red)", fontSize: 13 }}>{authError}</p>}
      </div>
    );
  }

  const graphFocused = tab === "graph" && gView === "graph" && gFocusMode;

  return (
    <div className={`admin-page${tab === "graph" ? " graph-wide" : ""}`}>
      <h1>KB 소비자보호 담당자 워크벤치</h1>

      {/* 티켓·피드백 통계는 그래프 탐색과 무관하다 — 캔버스에 세로를 내준다 */}
      {tab !== "graph" && (
      <div className="stat-row">
        <div className="stat-card">
          <div className="stat-num">{tickets.length}</div>
          <div>접수 티켓</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">{stats.feedback.up ?? 0} / {stats.feedback.down ?? 0}</div>
          <div>피드백 <IconThumbsUp size={12} /> / <IconThumbsDown size={12} /></div>
        </div>
        <div className="stat-card wide">
          {Object.entries(stats.categories).map(([cat, n]) => (
            <div key={cat} className="cat-bar">
              <span className="cat-name">{cat}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(n / totalCat) * 100}%` }} />
              </div>
              <span>{n}</span>
            </div>
          ))}
          {totalCat === 1 && Object.keys(stats.categories).length === 0 && <div>유형별 통계 없음</div>}
        </div>
      </div>
      )}

      <div className="mode-tabs">
        <button className={tab === "tickets" ? "active" : ""} onClick={() => setTab("tickets")}>민원 티켓</button>
        <button className={tab === "documents" ? "active" : ""} onClick={() => setTab("documents")}>지식베이스</button>
        <button className={tab === "logs" ? "active" : ""} onClick={() => setTab("logs")}>대화 로그</button>
        <button className={tab === "graph" ? "active" : ""} onClick={() => setTab("graph")}>지식 그래프</button>
        <button className={tab === "audit" ? "active" : ""} onClick={() => setTab("audit")}>감사 로그</button>
      </div>

      {tab === "audit" && (
        <table>
          <thead><tr><th>시각</th><th>행위자</th><th>행위</th><th>대상</th><th>IP</th></tr></thead>
          <tbody>
            {auditRows.map((a) => (
              <tr key={a.id}>
                <td className="mono">{a.created_at?.slice(0, 19).replace("T", " ")}</td>
                <td><b>{a.actor}</b></td>
                <td className="mono">{a.action}</td>
                <td className="summary-cell mono">{a.target}</td>
                <td className="mono">{a.ip}</td>
              </tr>
            ))}
            {auditRows.length === 0 && <tr><td colSpan={5}>감사 기록이 없습니다</td></tr>}
          </tbody>
        </table>
      )}

      {tab === "graph" && (
        <div className="graph-explorer">
          {gStats ? (
            <>
              {!graphFocused && (
              <div className="stat-row">
                <div className="stat-card"><div className="stat-num">{gStats.nodes}</div><div>노드</div></div>
                <div className="stat-card"><div className="stat-num">{gStats.edges}</div><div>엣지</div></div>
                <div className="stat-card"><div className="stat-num">{gStats.disabled_edges}</div><div>비활성 엣지</div></div>
                <div className="stat-card wide">
                  {Object.entries(gStats.edges_by_rel as Record<string, number>).map(([rel, n]) => (
                    <span key={rel} className="rel-chip">{REL_KO[rel] ?? rel} {n}</span>
                  ))}
                  <button className="rebuild-btn" disabled={gBusy} onClick={rebuildGraph}>
                    {gBusy ? "재구축 중…" : "그래프 재구축"}
                  </button>
                </div>
              </div>
              )}
              <div className="graph-toolbar">
                <div className="draft-form" style={{ maxWidth: 420, marginBottom: 0 }}>
                  <input placeholder="노드 검색 (예: 제19조, 사모펀드, 보이스피싱)"
                         value={gQuery} onChange={(e) => setGQuery(e.target.value)}
                         onKeyDown={(e) => e.key === "Enter" && graphSearch()} />
                  <button onClick={graphSearch}>검색</button>
                </div>
                <div className="graph-controls">
                  <button className={gView === "graph" ? "active" : ""}
                          onClick={() => setGView("graph")}>그래프</button>
                  <button className={gView === "table" ? "active" : ""}
                          onClick={() => setGView("table")}>표</button>
                  <span className="graph-sep" />
                  <label>
                    확장
                    <select value={gHops}
                            onChange={(e) => {
                              const h = Number(e.target.value);
                              setGHops(h);
                              loadSubgraph(gFocusId ?? undefined, h).catch(() => {});
                            }}>
                      <option value={1}>1단계</option>
                      <option value={2}>2단계</option>
                    </select>
                  </label>
                  <button onClick={() => loadSubgraph(undefined, gHops).catch(() => {})}>
                    전체 개관
                  </button>
                  {gView === "graph" && (
                    <>
                      <span className="graph-sep" />
                      <button className={gFocusMode ? "active" : ""}
                              onClick={() => setGFocusMode(!gFocusMode)}>
                        {gFocusMode ? "요약 보기" : "크게 보기"}
                      </button>
                    </>
                  )}
                </div>
              </div>

              {gView === "graph" && (
                <div className="graph-relfilter">
                  <span className="grf-label">관계 표시</span>
                  {Object.keys(gStats.edges_by_rel as Record<string, number>).map((rel) => {
                    const on = !gHidden.includes(rel);
                    return (
                      <button
                        key={rel}
                        className={on ? "on" : ""}
                        onClick={() => {
                          const next = on ? [...gHidden, rel] : gHidden.filter((r) => r !== rel);
                          setGHidden(next);
                          loadSubgraph(gFocusId ?? undefined, gHops, next).catch(() => {});
                        }}
                      >
                        {REL_KO[rel] ?? rel}
                      </button>
                    );
                  })}
                </div>
              )}

              {gView === "graph" && gGraph && (
                <div className={`graph-pane${graphFocused ? " focus" : ""}`}>
                  <Suspense fallback={<div className="graph-canvas" />}>
                    <GraphCanvas
                      nodes={gGraph.nodes}
                      edges={gGraph.edges}
                      focusId={gFocusId}
                      onNodeClick={(id) => openNode(id)}
                      onEdgeClick={(e) => setGEdge(e)}
                    />
                  </Suspense>
                  <div className="graph-side">
                    <div className="graph-legend">
                      {NODE_TYPE_LEGEND.map((l) => (
                        <span key={l.type}>
                          <i style={{ background: l.color }} />{l.label}
                        </span>
                      ))}
                    </div>
                    <p className="graph-hint">
                      노드를 누르면 그 노드를 중심으로 다시 그립니다. 엣지를 누르면
                      추출 근거(provenance)를 확인하고 비활성화할 수 있어요.
                      점선·붉은 엣지는 이미 비활성화된 연결입니다.
                    </p>
                    {gEdge ? (
                      <div className="graph-edge-detail">
                        <div className="ged-rel">{REL_KO[gEdge.rel] ?? gEdge.rel}</div>
                        <div className="ged-pair">
                          <b>{gGraph.nodes.find((n) => n.id === gEdge.src)?.label}</b>
                          {" → "}
                          <b>{gGraph.nodes.find((n) => n.id === gEdge.dst)?.label}</b>
                        </div>
                        <dl>
                          <dt>추출 방법</dt>
                          <dd className="mono">{gEdge.provenance?.method ?? "—"}</dd>
                          <dt>근거 스니펫</dt>
                          <dd>{gEdge.provenance?.evidence?.slice(0, 220) || "—"}</dd>
                        </dl>
                        <button className="link-btn" onClick={() => toggleEdge(gEdge.id)}>
                          {gEdge.disabled ? "복원" : "비활성화"}
                        </button>
                      </div>
                    ) : (
                      <div className="graph-edge-detail empty">엣지를 선택하면 provenance가 표시됩니다.</div>
                    )}
                  </div>
                </div>
              )}
              {gNodes.length > 0 && !gFocus && (
                <table>
                  <thead><tr><th>타입</th><th>노드</th><th></th></tr></thead>
                  <tbody>
                    {gNodes.map((n) => (
                      <tr key={n.id}>
                        <td>{n.type}</td><td>{n.label}</td>
                        <td><button className="link-btn" onClick={() => openNode(n.id)}>이 노드 보기</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {gView === "table" && gFocus && (
                <>
                  <p style={{ margin: "8px 0" }}>
                    <button className="link-btn" onClick={() => setGFocus(null)}>← 검색 결과</button>
                    &nbsp; <b>{gFocus.node.label}</b> <span className="mono">({gFocus.node.type})</span>
                  </p>
                  <table>
                    <thead><tr><th>관계</th><th>연결 노드</th><th>추출 방법</th><th>근거 스니펫</th><th>상태</th></tr></thead>
                    <tbody>
                      {gFocus.neighbors.map((nb: any) => (
                        <tr key={nb.edge.id} style={nb.disabled ? { opacity: 0.45 } : undefined}>
                          <td>{REL_KO[nb.edge.rel] ?? nb.edge.rel}</td>
                          <td><button className="link-btn" onClick={() => openNode(nb.node.id)}>{nb.node.label}</button></td>
                          <td className="mono">{nb.edge.provenance?.method}</td>
                          <td className="summary-cell">{nb.edge.provenance?.evidence?.slice(0, 80)}</td>
                          <td>
                            <button className="link-btn" onClick={() => toggleEdge(nb.edge.id)}>
                              {nb.disabled ? "복원" : "비활성화"}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </>
          ) : (
            <p>그래프가 없습니다. 서버에서 <code>python scripts/build_graph.py</code>를 실행하세요.</p>
          )}
        </div>
      )}

      {tab === "tickets" && (
        <table>
          <thead>
            <tr><th>ID</th><th>세션</th><th>유형</th><th>요약</th><th>상태</th><th>접수일</th></tr>
          </thead>
          <tbody>
            {tickets.map((t) => (
              <tr key={t.id}>
                <td>{t.id}</td>
                <td className="mono">{t.session_id.slice(0, 14)}…</td>
                <td>{t.category}</td>
                <td className="summary-cell">{t.summary}</td>
                <td><span className="tag partial">{t.status}</span></td>
                <td>{t.created_at?.slice(0, 10)}</td>
              </tr>
            ))}
            {tickets.length === 0 && <tr><td colSpan={6}>접수된 티켓이 없습니다</td></tr>}
          </tbody>
        </table>
      )}

      {tab === "documents" && (
        <>
          <div className="doc-filter">
            {["", "laws", "cases", "interpretations", "procedures", "kb_products"].map((k) => (
              <button key={k} className={docKind === k ? "active" : ""} onClick={() => setDocKind(k)}>
                {k === "" ? "전체" : k === "laws" ? "법령" : k === "cases" ? "사례" : k === "interpretations" ? "유권해석" : k === "procedures" ? "절차" : "KB 상품"}
              </button>
            ))}
            <span className="doc-count">{docs.length}건 · 전부 실제 출처 문서</span>
          </div>
          <table>
            <thead>
              <tr><th>종류</th><th>제목</th><th>출처</th><th>수집일</th></tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.id}>
                  <td>{d.kind}</td>
                  <td>{d.title}</td>
                  <td><a href={d.source_url} target="_blank" rel="noreferrer">{d.source.slice(0, 40)}</a></td>
                  <td>{d.ingested_at?.slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {tab === "logs" && (
        <table>
          <thead>
            <tr><th>세션</th><th>역할</th><th>내용</th><th>시각</th></tr>
          </thead>
          <tbody>
            {logs.map((m) => (
              <tr key={m.id}>
                <td className="mono">{m.session_id.slice(0, 14)}…</td>
                <td>{m.role === "user" ? <><IconUser size={13} /> 소비자</> : <><IconBot size={13} /> 에이전트</>}</td>
                <td className="summary-cell">{m.content.slice(0, 120)}</td>
                <td>{m.created_at?.slice(11, 19)}</td>
              </tr>
            ))}
            {logs.length === 0 && <tr><td colSpan={4}>대화 로그가 없습니다</td></tr>}
          </tbody>
        </table>
      )}
    </div>
  );
}
