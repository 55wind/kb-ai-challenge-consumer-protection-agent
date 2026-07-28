// 프로덕션(FastAPI 정적 서빙)은 동일 오리진, 개발(vite dev)만 로컬 백엔드 지정
const BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

export interface Citation {
  record_id: string;
  title: string;
  source: string;
  source_url: string;
  quote: string;
}

export interface CaseSummary {
  id: string;
  title: string;
  decision: string;
  key_reasoning: string;
  source: string;
  source_url: string;
}

// check = 룰에는 걸렸으나 근거 조문을 찾지 못해 위반 판정을 보류한 상태
export type ScanLevel = "red" | "yellow" | "check" | "green";

export interface ScanItem {
  clause: string;
  level: ScanLevel;
  rule: string;
  advice: string;
  law_citation: Citation | null;
}

export interface ScanSummary {
  total: number;
  red: number;
  yellow: number;
  check: number;
  unparsed: number;
  top_risk: string;
  actions: string[];
}

export interface DeadlineItem {
  action: string;
  period: string;
  computed_deadline: string | null;
  urgency: "expired" | "urgent" | "ok" | "unknown";
  urgency_note: string;
  law_citation: Citation | null;
}

export interface EvidenceItem {
  seq: number;
  timestamp: string;
  speaker: string;
  utterance: string;
  level: "red" | "yellow" | "check";
  rule: string;
  reason: string;
  law_citation: Citation | null;
  related_cases: CaseSummary[];
}

export interface GraphNode {
  id: string;
  type: "category" | "law" | "case" | "procedure" | "product" | "rule" | "deadline";
  label: string;
  meta: Record<string, any>;
}

export interface GraphEdge {
  id: string;
  src: string;
  rel: string;
  dst: string;
  provenance: Record<string, any>;
}

export interface EvidenceGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export async function createSession(): Promise<string> {
  const resp = await fetch(`${BASE}/api/session`, { method: "POST" });
  if (!resp.ok) throw new Error(`세션 생성 실패 (${resp.status})`);
  return (await resp.json()).session_id;
}

export type AgentEventHandler = (event: string, payload: any) => void;

export async function chatStream(
  sessionId: string,
  text: string,
  mode: "chat" | "scan",
  onEvent: AgentEventHandler,
): Promise<void> {
  const resp = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, text, mode }),
  });
  if (!resp.ok || !resp.body) throw new Error(`서버 오류 (${resp.status})`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\r\n\r\n").join("\n\n").split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      let event = "message";
      let data = "";
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) {
        try {
          onEvent(event, JSON.parse(data));
        } catch {
          /* keep-alive 등 무시 */
        }
      }
    }
  }
}

export async function requestDraft(sessionId: string, respondent: string) {
  const resp = await fetch(`${BASE}/api/draft/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ respondent }),
  });
  if (!resp.ok) throw new Error("초안 생성 실패");
  return resp.json() as Promise<{ markdown: string; evidence: string[] }>;
}

// 세션을 함께 보낸다 — 서버가 "이 답변을 만든 세션인지"로 소유권을 확인한다
export async function sendFeedback(sessionId: string, messageId: number, vote: "up" | "down") {
  await fetch(`${BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message_id: messageId, vote }),
  });
}

async function adminGet(path: string, token: string) {
  const resp = await fetch(`${BASE}${path}`, { headers: { "X-Admin-Token": token } });
  if (resp.status === 401) throw new Error("UNAUTHORIZED");
  return resp.json();
}

async function adminPost(path: string, token: string, body?: any) {
  const resp = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "X-Admin-Token": token, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (resp.status === 401) throw new Error("UNAUTHORIZED");
  return resp.json();
}

export const admin = {
  tickets: (token: string) => adminGet("/api/admin/tickets", token),
  stats: (token: string) => adminGet("/api/admin/stats", token),
  documents: (token: string, kind?: string) =>
    adminGet(`/api/admin/documents${kind ? `?kind=${kind}` : ""}`, token),
  logs: (token: string, sessionId?: string) =>
    adminGet(`/api/admin/logs${sessionId ? `?session_id=${sessionId}` : ""}`, token),
  graphStats: (token: string) => adminGet("/api/admin/graph/stats", token),
  graphSearch: (token: string, q: string) =>
    adminGet(`/api/admin/graph/search?q=${encodeURIComponent(q)}`, token),
  graphNeighbors: (token: string, id: string) =>
    adminGet(`/api/admin/graph/neighbors?id=${encodeURIComponent(id)}`, token),
  graphSubgraph: (token: string, id?: string, hops = 1, exclude: string[] = []) =>
    adminGet(
      `/api/admin/graph/subgraph?hops=${hops}` +
        (id ? `&id=${encodeURIComponent(id)}` : "") +
        (exclude.length ? `&exclude=${encodeURIComponent(exclude.join(","))}` : ""),
      token,
    ),
  graphToggleEdge: (token: string, edgeId: string, reason?: string) =>
    adminPost(`/api/admin/graph/edge/${edgeId}/toggle`, token, { reason: reason ?? "" }),
  graphRebuild: (token: string) => adminPost("/api/admin/graph/rebuild", token),
  audit: (token: string) => adminGet("/api/admin/audit", token),
};
