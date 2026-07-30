// 프로덕션(FastAPI 정적 서빙)은 동일 오리진, 개발(vite dev)만 로컬 백엔드 지정
const BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

/**
 * 통신 실패를 사람이 읽을 수 있는 안내로 바꾼다.
 *
 * fetch가 실패하면 브라우저는 "TypeError: Failed to fetch"(크롬) 또는
 * "NetworkError when attempting to fetch resource"(파이어폭스)를 던진다.
 * 이것을 그대로 화면에 띄우면 취약 소비자는 무엇을 해야 할지 알 수 없다.
 * 답변이 오다 끊긴 경우까지 포함해 같은 문구로 안내한다.
 */
export class ConnectionLost extends Error {
  constructor() {
    super(
      "서버와 연결이 끊어졌습니다. 잠시 후 다시 시도해 주세요. " +
        "계속 반복되면 금융감독원 1332로 문의하실 수 있습니다.",
    );
    this.name = "ConnectionLost";
  }
}

/**
 * 화면에 띄울 문구. 예외 원문(`TypeError: Failed to fetch`)을 그대로 보여주지
 * 않는다 — 이 서비스의 사용자는 개발자가 아니다.
 */
export function humanError(e: unknown): string {
  if (e instanceof ConnectionLost) return e.message;
  if (isNetworkFailure(e)) return new ConnectionLost().message;
  if (e instanceof Error && e.message) return e.message;
  return "요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.";
}

function isNetworkFailure(e: unknown): boolean {
  // 서버가 보낸 오류(`서버 오류 (500)`)는 그대로 두고, fetch 자체가 실패한
  // 경우만 갈아낀다. 브라우저마다 메시지가 달라 타입으로 판별한다.
  return e instanceof TypeError || (e instanceof DOMException && e.name === "AbortError");
}

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
  // 분류 2순위에서 온 카드. 확정된 기한과 섞어 보이면 화면이 사실을 과장한다.
  tentative?: boolean;
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
  let resp: Response;
  try {
    resp = await fetch(`${BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, text, mode }),
    });
  } catch (e) {
    if (isNetworkFailure(e)) throw new ConnectionLost();
    throw e;
  }
  if (!resp.ok || !resp.body) throw new Error(`서버 오류 (${resp.status})`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    // 답변이 오는 도중에도 끊길 수 있다. CPU 추론은 생성이 수십 초라
    // 이 구간이 길고, 그만큼 끊길 여지도 크다.
    let chunk: ReadableStreamReadResult<Uint8Array>;
    try {
      chunk = await reader.read();
    } catch (e) {
      if (isNetworkFailure(e)) throw new ConnectionLost();
      throw e;
    }
    const { done, value } = chunk;
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
