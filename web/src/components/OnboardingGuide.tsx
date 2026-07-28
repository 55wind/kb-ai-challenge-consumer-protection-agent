import { useEffect, useLayoutEffect, useState } from "react";

type Step = { target?: string; title: string; body: string; consent?: boolean };

// 개인정보 수집·이용 고지 (은행 서비스 전제조건 — 수집 항목·목적·보유기간을 명시)
const CONSENT_ROWS: [string, string][] = [
  ["수집 항목", "상담 내용, 붙여넣은 계약·문자·녹취 텍스트 (연락처·주민번호·계좌번호는 저장 전 자동 마스킹)"],
  ["이용 목적", "상담 답변 생성, 불공정 조항 검사, 분쟁조정 신청서 초안 작성, 소비자보호 담당자의 민원 처리"],
  ["보유 기간", "상담 종료 후 2시간 경과 시 세션 자동 삭제. 민원 티켓은 처리 완료 후 은행 민원처리 규정에 따름"],
  ["처리 위치", "은행 내부 서버에서만 처리. 외부 AI API로 전송하지 않음"],
];

const STEPS: Step[] = [
  {
    title: "금융 소비자 보호 에이전트예요",
    body: "어려운 금융 문제, 혼자 고민하지 마세요.\n어떻게 쓰는지 30초만 알려드릴게요.",
  },
  {
    target: '[data-tour="chat-tab"]',
    title: "① 편하게 상담하세요",
    body: "궁금한 점이나 억울한 일을 평소 말하듯 적으면 돼요. 법 용어를 몰라도 괜찮아요. 답변마다 법령·실제 사례 근거를 함께 보여드려요.",
  },
  {
    target: '[data-tour="scan-tab"]',
    title: "② 불공정 조항 검사",
    body: "계약서 조항, 광고 문자, 상담 녹취록을 그대로 붙여넣어 보세요. 위험한 부분을 자동으로 찾아 색깔로 표시해 드려요. 부모님이 받은 문자도 대신 검사할 수 있어요.",
  },
  {
    target: '[data-tour="easy"]',
    title: "③ 쉬운말 풀이",
    body: "켜 두면 어려운 금융 용어가 나올 때마다 쉬운 말 설명을 함께 보여드려요. 기본으로 켜져 있어요.",
  },
  {
    target: '[data-tour="composer"]',
    title: "④ 여기에 입력하세요",
    body: "입력하고 Enter 키(또는 노란 버튼)를 누르면 시작돼요. 상담이 끝나면 분쟁조정 신청서 초안까지 만들어 드려요.",
  },
  {
    title: "시작 전에 알려드려요",
    body:
      "· 이 서비스는 법률 자문이 아닌 정보 제공이에요. 정확한 판단은 금융감독원(1332)이나 전문가와 상담하세요.\n" +
      "· 상담 내용은 은행 내부에만 저장되며 외부로 전송되지 않아요.",
    consent: true,
  },
];

const SPOT_PAD = 6;
const CARD_W = 320;
const CARD_GAP = 14;

export default function OnboardingGuide({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [idx, setIdx] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const step = STEPS[idx];
  const last = idx === STEPS.length - 1;

  useEffect(() => {
    if (open) setIdx(0);
  }, [open]);

  useLayoutEffect(() => {
    if (!open) return;
    const update = () => {
      if (!step.target) {
        setRect(null);
        return;
      }
      const el = document.querySelector(step.target);
      setRect(el ? el.getBoundingClientRect() : null);
    };
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [open, idx, step.target]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  let cardStyle: React.CSSProperties;
  if (!rect) {
    cardStyle = { top: "50%", left: "50%", transform: "translate(-50%, -50%)" };
  } else {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const left = Math.min(Math.max(16, rect.left), Math.max(16, vw - CARD_W - 16));
    const belowTop = rect.bottom + SPOT_PAD + CARD_GAP;
    if (belowTop + 230 > vh) {
      cardStyle = { left, bottom: vh - rect.top + SPOT_PAD + CARD_GAP };
    } else {
      cardStyle = { left, top: belowTop };
    }
  }

  return (
    <div className="tour" role="dialog" aria-modal="true" aria-label="사용법 안내">
      {rect ? (
        <div
          className="tour-spotlight"
          style={{
            top: rect.top - SPOT_PAD,
            left: rect.left - SPOT_PAD,
            width: rect.width + SPOT_PAD * 2,
            height: rect.height + SPOT_PAD * 2,
          }}
        />
      ) : (
        <div className="tour-dim" />
      )}
      <div key={idx} className="tour-card" style={cardStyle}>
        <div className="tour-badge">사용법 안내 {idx + 1} / {STEPS.length}</div>
        <div className="tour-title">{step.title}</div>
        <div className="tour-body">{step.body}</div>
        {step.consent && (
          <details className="tour-consent">
            <summary>개인정보 수집·이용 안내 (자세히)</summary>
            <dl>
              {CONSENT_ROWS.map(([label, value]) => (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
          </details>
        )}
        <div className="tour-dots" aria-hidden="true">
          {STEPS.map((_, i) => (
            <span key={i} className={i === idx ? "on" : ""} />
          ))}
        </div>
        <div className="tour-actions">
          <button className="tour-skip" onClick={onClose}>건너뛰기</button>
          <div className="tour-nav">
            {idx > 0 && (
              <button className="tour-prev" onClick={() => setIdx(idx - 1)}>이전</button>
            )}
            <button
              className="tour-next"
              autoFocus
              onClick={() => (last ? onClose() : setIdx(idx + 1))}
            >
              {last ? "확인했어요" : "다음"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
