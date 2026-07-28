import { useEffect, useRef, useState } from "react";
import {
  chatStream,
  createSession,
  requestDraft,
  sendFeedback,
  type CaseSummary,
  type Citation,
  type DeadlineItem,
  type EvidenceItem,
  type ScanItem,
  type ScanSummary,
} from "../lib/api";
import OnboardingGuide from "../components/OnboardingGuide";
import Markdown from "../components/Markdown";
import {
  IconAlarmClock,
  IconChart,
  IconChat,
  IconCheck,
  IconFileText,
  IconHourglass,
  IconMic,
  IconPaperclip,
  IconPen,
  IconScale,
  IconSearch,
  IconThumbsDown,
  IconThumbsUp,
} from "../components/icons";

type Block =
  | { kind: "user"; text: string }
  | { kind: "status"; text: string }
  | { kind: "answer"; text: string; messageId?: number; voted?: string; streaming?: boolean }
  | { kind: "citations"; items: Citation[] }
  | { kind: "glossary"; items: { term: string; definition: string }[] }
  | { kind: "outlook"; stats: Record<string, number>; commentary: string; caveat?: string; similar: CaseSummary[] }
  | { kind: "scan"; items: ScanItem[]; redCount: number; inputKind?: string; summary?: ScanSummary | null }
  | { kind: "deadlines"; items: DeadlineItem[] }
  | { kind: "transcript"; items: EvidenceItem[]; summary: ScanSummary | null; notice?: string; scope?: string; unparsed?: number }
  | { kind: "draftOffer"; message: string }
  | { kind: "notice"; text: string; tone: "warn" | "error" };

const LEVEL_LABEL = { red: "위험", yellow: "주의", check: "확인 필요", green: "정상" } as const;

// 무엇을 검사했는지 밝힌다. 화자 판별 결과에 따라 검사 대상이 달라지므로,
// 이걸 감추면 "직원 발언만 봤는지 전체를 봤는지" 사용자가 알 수 없다.
// 슬롯 키 → 사람이 읽는 이름 (app/agent/slots.py의 SLOT_LABEL과 같은 뜻)
const SLOT_LABEL: Record<string, string> = {
  who: "누가", when: "언제", product: "상품", incident: "무슨 일", demand: "요구사항",
};

const SCOPE_LABEL: Record<string, string> = {
  staff: "판매 직원의 발언만 검사했습니다.",
  mixed: "직원으로 확인된 발언을 검사했습니다. 역할이 불분명한 발언도 함께 검사했습니다.",
  all: "누가 판매 직원인지 확정할 수 없어 모든 발언을 검사했습니다.",
  customer_only: "직원 발언을 찾지 못해 모든 발언을 검사했습니다.",
};
// 결정 라벨 → 색. 3종 밖 라벨(각하·조정성립 등)을 초록(인용)으로 칠하면
// 각하를 승소로 보여주게 된다. 모르는 라벨은 중립색으로 둔다.
const DECISION_TAG: Record<string, string> = {
  "인용": "grant", "일부인용": "partial", "기각": "reject",
};
const decisionTag = (d: string) => DECISION_TAG[d] ?? "neutral";
const URGENCY_LABEL = { urgent: "긴급", ok: "여유", unknown: "확인 필요", expired: "기한 경과 가능" } as const;
const KIND_LABEL: Record<string, string> = {
  transcript: "녹취록으로 인식했습니다",
  sms_batch: "문자함으로 인식했습니다",
  document: "문서로 인식했습니다",
};

export default function Chat() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [blocks, setBlocks] = useState<Block[]>([]);
  // 사실관계 수집 진행 — 무엇이 채워졌고 무엇이 남았는지
  const [slots, setSlots] = useState<{ filled: string[]; missing: string[] } | null>(null);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"chat" | "scan">("chat");
  const [busy, setBusy] = useState(false);
  // 면책 고지는 서버가 첫 응답에서 다시 확인해 주지만, 최초 진입 화면부터 보여야 한다 (M-8)
  const [disclaimer, setDisclaimer] = useState(
    "본 서비스는 법률 자문이 아닌 정보 제공입니다. 정확한 판단은 금융감독원(1332) 또는 전문가와 상담하세요.",
  );
  const [draftMd, setDraftMd] = useState("");
  const [respondent, setRespondent] = useState("");
  const [showDraftForm, setShowDraftForm] = useState(false);
  const [easyMode, setEasyMode] = useState(true);
  const [showGuide, setShowGuide] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!sessionStorage.getItem("kbGuideSeen")) {
      sessionStorage.setItem("kbGuideSeen", "1");
      setShowGuide(true);
    }
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [blocks, draftMd]);

  useEffect(() => {
    createSession()
      .then(setSessionId)
      .catch(() => push({ kind: "notice", text: "서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해 주세요.", tone: "error" }));
  }, []);

  const push = (b: Block) => setBlocks((prev) => [...prev, b]);

  // 누른 표시를 남긴다. 없으면 같은 답변에 몇 번이고 다시 누르게 되고,
  // 눌렀는지조차 알 수 없다.
  function vote(index: number, messageId: number, v: "up" | "down") {
    if (!sessionId) return;
    setBlocks((prev) => prev.map((b, i) =>
      i === index && b.kind === "answer" ? { ...b, voted: v } : b));
    sendFeedback(sessionId, messageId, v).catch(() => {
      setBlocks((prev) => prev.map((b, i) =>
        i === index && b.kind === "answer" ? { ...b, voted: undefined } : b));
    });
  }

  async function send() {
    const text = input.trim();
    if (!text || busy || !sessionId) return;
    setInput("");
    setBusy(true);
    push({ kind: "user", text });
    try {
      await chatStream(sessionId, text, mode, (event, payload) => {
        setBlocks((prev) => prev.filter((b) => b.kind !== "status"));
        switch (event) {
          case "disclaimer":
            setDisclaimer(payload.message);
            break;
          case "status":
            push({ kind: "status", text: payload.message });
            break;
          case "answer_delta":
            // 생성 중 조각. 아직 근거 검증 전이라 '작성 중'으로만 보여준다.
            setBlocks((prev) => {
              const last = prev[prev.length - 1];
              if (last?.kind === "answer" && last.streaming) {
                return [...prev.slice(0, -1),
                        { ...last, text: last.text + payload.text }];
              }
              return [...prev, { kind: "answer", text: payload.text, streaming: true }];
            });
            break;
          case "answer_discard":
            // 근거 인용이 없어 답변으로 확정하지 못했다 — 잠정 표시를 지운다
            setBlocks((prev) => prev.filter((b) => !(b.kind === "answer" && b.streaming)));
            break;
          case "answer":
            // 검증을 통과했다. 잠정 표시가 있으면 확정본으로 교체한다.
            setBlocks((prev) => {
              const i = prev.findIndex((b) => b.kind === "answer" && b.streaming);
              const done: Block = { kind: "answer", text: payload.text,
                                    messageId: payload.message_id };
              return i >= 0 ? [...prev.slice(0, i), done, ...prev.slice(i + 1)]
                            : [...prev, done];
            });
            break;
          case "citations":
            push({ kind: "citations", items: payload.items });
            break;
          case "slots":
            setSlots({
              filled: Object.keys(payload.filled ?? {}),
              missing: payload.missing ?? [],
            });
            break;
          case "glossary":
            push({ kind: "glossary", items: payload.items });
            break;
          case "outlook":
            push({ kind: "outlook", stats: payload.stats, commentary: payload.commentary,
                   caveat: payload.caveat, similar: payload.similar });
            break;
          case "scan_report":
            push({ kind: "scan", items: payload.items, redCount: payload.red_count,
                   inputKind: payload.input_kind, summary: payload.summary });
            break;
          case "deadlines":
            push({ kind: "deadlines", items: payload.items });
            break;
          case "transcript_report":
            push({ kind: "transcript", items: payload.items, summary: payload.summary,
                   notice: payload.notice, scope: payload.scope, unparsed: payload.unparsed });
            break;
          case "busy":
            push({ kind: "status", text: payload.message });
            break;
          case "draft_offer":
            push({ kind: "draftOffer", message: payload.message });
            setShowDraftForm(true);
            break;
          case "no_evidence":
            push({ kind: "notice", text: payload.message, tone: "warn" });
            break;
          case "error":
            push({ kind: "notice", text: payload.message, tone: "error" });
            break;
        }
      });
    } catch (e) {
      push({ kind: "notice", text: String(e), tone: "error" });
    } finally {
      setBlocks((prev) => prev.filter((b) => b.kind !== "status"));
      setBusy(false);
    }
  }

  async function makeDraft() {
    if (!sessionId) return;
    setBusy(true);
    try {
      const d = await requestDraft(sessionId, respondent);
      setDraftMd(d.markdown);
      setShowDraftForm(false);
    } catch (e) {
      push({ kind: "notice", text: String(e), tone: "error" });
    } finally {
      setBusy(false);
    }
  }

  function downloadDraft() {
    const blob = new Blob([draftMd], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "분쟁조정신청서_초안.md";
    a.click();
  }

  return (
    <div className="chat-page">
      <OnboardingGuide open={showGuide} onClose={() => setShowGuide(false)} />
      {disclaimer && <div className="disclaimer"><IconScale size={15} /><span>{disclaimer}</span></div>}
      <div className="mode-tabs">
        <button data-tour="chat-tab" className={mode === "chat" ? "active" : ""} onClick={() => setMode("chat")}>
          <IconChat size={15} /> 상담하기
        </button>
        <button data-tour="scan-tab" className={mode === "scan" ? "active" : ""} onClick={() => setMode("scan")}>
          <IconSearch size={15} /> 불공정 조항 검사
        </button>
        <label className="easy-toggle" data-tour="easy">
          <input type="checkbox" checked={easyMode} onChange={(e) => setEasyMode(e.target.checked)} />
          쉬운말 풀이
        </label>
        <button className="help-btn" onClick={() => setShowGuide(true)} aria-label="사용법 다시 보기">
          ? 사용법
        </button>
      </div>

      <div className="messages">
        {blocks.length === 0 && (
          <div className="empty-hint">
            {mode === "chat"
              ? "금융 관련 궁금한 점이나 억울한 일을 편하게 말씀해 주세요.\n예) “은행에서 펀드 가입할 때 위험 설명을 못 들었는데 손실이 났어요.”"
              : "상품설명서·계약 조항, 문자 여러 건, 텍스트로 옮긴 판매 상담 녹취록을 붙여넣으면 자동으로 인식해 검사해 드립니다.\n부모님이 받은 문자를 대신 검사해 드릴 수 있어요."}
          </div>
        )}
        {blocks.map((b, i) => {
          switch (b.kind) {
            case "user":
              return <div key={i} className="bubble user">{b.text}</div>;
            case "status":
              return <div key={i} className="status"><IconHourglass size={13} /> {b.text}</div>;
            case "answer":
              return (
                <div key={i} className={`bubble bot${b.streaming ? " streaming" : ""}`}>
                  {b.text}
                  {b.streaming && (
                    <span className="streaming-tag">작성 중 — 근거 확인 전입니다</span>
                  )}
                  {b.messageId && (
                    <span className="votes">
                      <button
                        className={b.voted === "up" ? "voted" : ""}
                        aria-pressed={b.voted === "up"}
                        onClick={() => vote(i, b.messageId!, "up")}
                        aria-label="도움이 됐어요"
                      ><IconThumbsUp size={15} /></button>
                      <button
                        className={b.voted === "down" ? "voted" : ""}
                        aria-pressed={b.voted === "down"}
                        onClick={() => vote(i, b.messageId!, "down")}
                        aria-label="도움이 안 됐어요"
                      ><IconThumbsDown size={15} /></button>
                    </span>
                  )}
                </div>
              );
            case "citations":
              return (
                <div key={i} className="citations">
                  {b.items.map((c, j) => (
                    <div key={j} className="citation-card">
                      <div className="citation-title"><IconFileText size={14} /> [{j + 1}] {c.title}</div>
                      <div className="citation-quote">{c.quote}…</div>
                      <a href={c.source_url} target="_blank" rel="noreferrer">{c.source}</a>
                    </div>
                  ))}
                </div>
              );
            case "glossary":
              return easyMode ? (
                <div key={i} className="glossary">
                  {b.items.map((g, j) => (
                    <div key={j}><b>{g.term}</b> — {g.definition}</div>
                  ))}
                </div>
              ) : null;
            case "outlook":
              return (
                <div key={i} className="outlook">
                  <div className="outlook-stats">
                    유사한 실제 조정 사례 {b.similar.length}건 —
                    <span className="tag grant">인용 {b.stats["인용"]}</span>
                    <span className="tag partial">일부인용 {b.stats["일부인용"]}</span>
                    <span className="tag reject">기각 {b.stats["기각"]}</span>
                    {b.stats["기타"] > 0 && (
                      <span className="tag neutral">기타 {b.stats["기타"]}</span>
                    )}
                  </div>
                  <Markdown source={b.commentary} className="outlook-commentary" />
                  {b.caveat && <div className="outlook-caveat"><IconChart size={14} /><span>{b.caveat}</span></div>}
                  {b.similar.map((s, j) => (
                    <div key={j} className="citation-card">
                      <div className="citation-title">
                        <span className={`tag ${decisionTag(s.decision)}`}>{s.decision}</span> {s.title}
                      </div>
                      <div className="citation-quote">{s.key_reasoning.slice(0, 150)}…</div>
                      <a href={s.source_url} target="_blank" rel="noreferrer">{s.source}</a>
                    </div>
                  ))}
                </div>
              );
            case "deadlines":
              return (
                <div key={i} className="deadlines">
                  <div className="deadlines-head"><IconAlarmClock size={16} /> 지금 해야 할 일</div>
                  {b.items.map((d, j) => (
                    <div key={j} className={`deadline-item ${d.urgency}`}>
                      <span className={`tag urgency-${d.urgency}`}>{URGENCY_LABEL[d.urgency]}</span>
                      <div>
                        <div className="deadline-action">{d.action}</div>
                        <div className="deadline-period">
                          {d.period}
                          {d.computed_deadline && ` · 기한(말씀하신 시기 기준 추정): ${d.computed_deadline}`}
                        </div>
                        <div className="scan-advice">{d.urgency_note}</div>
                        {d.law_citation && (
                          <a href={d.law_citation.source_url} target="_blank" rel="noreferrer">
                            근거: {d.law_citation.title}
                          </a>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              );
            case "transcript":
              return (
                <div key={i} className="scan-report">
                  {b.summary && (
                    <div className="scan-summary">
                      <IconMic size={14} /> 녹취 발언 {b.summary.total}건 중 위험 {b.summary.red}건 · 주의 {b.summary.yellow}건
                      {b.summary.check > 0 && ` · 확인 필요 ${b.summary.check}건`}
                    </div>
                  )}
                  {b.scope && SCOPE_LABEL[b.scope] && (
                    <div className="scan-scope">{SCOPE_LABEL[b.scope]}</div>
                  )}
                  {b.notice && <div className="scan-notice">{b.notice}</div>}
                  {!!b.unparsed && b.unparsed > 0 && (
                    <div className="scan-notice">
                      화자를 구분하지 못한 줄 {b.unparsed}건은 검사에서 빠졌습니다. 해당 줄은 위험 여부를 확인하지 못했습니다.
                    </div>
                  )}
                  {b.items.map((it, j) => (
                    <div key={j} className={`scan-item ${it.level}`}>
                      <span className="scan-level">{LEVEL_LABEL[it.level]}</span>
                      <div>
                        <div className="scan-clause">
                          {it.timestamp && <span className="mono">{it.timestamp} </span>}
                          {it.speaker}: “{it.utterance}”
                        </div>
                        <div className="scan-advice">{it.rule} — {it.reason}</div>
                        {it.law_citation && (
                          <a href={it.law_citation.source_url} target="_blank" rel="noreferrer">
                            근거: {it.law_citation.title}
                          </a>
                        )}
                        {it.related_cases.length > 0 && (
                          <details>
                            <summary>이 발언이 쟁점이 된 실제 사례 {it.related_cases.length}건</summary>
                            {it.related_cases.map((c, k) => (
                              <div key={k} className="citation-card">
                                <div className="citation-title">
                                  <span className={`tag ${c.decision === "기각" ? "reject" : "grant"}`}>{c.decision}</span> {c.title}
                                </div>
                                <a href={c.source_url} target="_blank" rel="noreferrer">{c.source}</a>
                              </div>
                            ))}
                          </details>
                        )}
                      </div>
                    </div>
                  ))}
                  {b.items.length > 0 && (
                    <button className="attach-evidence" onClick={() => setShowDraftForm(true)}>
                      <IconPaperclip size={15} /> 이 증거를 신청서에 첨부하기
                    </button>
                  )}
                </div>
              );
            case "scan":
              return (
                <div key={i} className="scan-report">
                  {b.inputKind && <div className="kind-chip">{KIND_LABEL[b.inputKind] ?? ""}</div>}
                  <div className="scan-summary">
                    {b.summary
                      ? `${b.inputKind === "sms_batch" ? "문자" : "항목"} ${b.summary.total}건 중 위험 ${b.summary.red}건 · 주의 ${b.summary.yellow}건`
                        + (b.summary.check > 0 ? ` · 확인 필요 ${b.summary.check}건` : "")
                      : `검사 완료 — 위험 ${b.redCount}건 발견`}
                  </div>
                  {b.summary?.top_risk && (
                    <div className="scan-toprisk">
                      <IconAlarmClock size={14} />
                      <span>가장 먼저 확인할 곳 — “{b.summary.top_risk}”</span>
                    </div>
                  )}
                  <div className="scan-caveat">
                    자동 검사 결과이며 법적 판단이 아닙니다. 위험·주의 표시는 확인이 필요한 지점을 알려드리는 것입니다.
                  </div>
                  {b.summary && b.summary.actions.length > 0 && (
                    <div className="scan-actions">
                      {b.summary.actions.map((a, k) => <div key={k}><IconCheck size={14} /><span>{a}</span></div>)}
                    </div>
                  )}
                  {b.items.map((it, j) => (
                    <div key={j} className={`scan-item ${it.level}`}>
                      <span className="scan-level">{LEVEL_LABEL[it.level]}</span>
                      <div>
                        <div className="scan-clause">{it.clause}</div>
                        {it.level !== "green" && (
                          <>
                            <div className="scan-advice">{it.advice}</div>
                            {it.law_citation && (
                              <a href={it.law_citation.source_url} target="_blank" rel="noreferrer">
                                근거: {it.law_citation.title}
                              </a>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              );
            case "draftOffer":
              return <div key={i} className="bubble bot offer">{b.message}</div>;
            case "notice":
              return <div key={i} className={`notice ${b.tone}`}>{b.text}</div>;
          }
        })}
        {slots && slots.missing.length > 0 && (
          <div className="slot-progress">
            <IconCheck size={13} />
            <span>
              사실관계 {slots.filled.length}/{slots.filled.length + slots.missing.length}개 정리됨 ·
              남은 항목: {slots.missing.map((k) => SLOT_LABEL[k] ?? k).join(" · ")}
            </span>
          </div>
        )}
        {showDraftForm && (
          <div className="draft-form">
            <input
              placeholder="금융회사명 (예: ○○은행)"
              value={respondent}
              onChange={(e) => setRespondent(e.target.value)}
            />
            <button disabled={busy} onClick={makeDraft}>신청서 초안 만들기</button>
          </div>
        )}
        {draftMd && (
          <div className="draft-preview">
            <div className="draft-head">
              <span className="draft-title"><IconPen size={15} /> 분쟁조정 신청서 초안</span>
              <button onClick={downloadDraft}>다운로드 (.md)</button>
            </div>
            <div className="draft-body">
              <Markdown source={draftMd} />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="composer" data-tour="composer">
        <textarea
          value={input}
          placeholder={mode === "chat" ? "궁금한 점이나 겪으신 일을 편하게 적어 주세요" : "검사할 문서·문자·녹취록을 붙여넣어 주세요"}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button disabled={busy || !input.trim()} onClick={send}>
          {busy ? "확인하고 있어요" : mode === "chat" ? "물어보기" : "검사하기"}
        </button>
      </div>
    </div>
  );
}
