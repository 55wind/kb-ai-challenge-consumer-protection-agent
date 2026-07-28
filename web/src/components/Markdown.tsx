// 의존성 없는 최소 마크다운 렌더러.
// LLM·템플릿이 만드는 범위(제목·인용·목록·체크리스트·표·구분선·굵게)만 다룬다.
// dangerouslySetInnerHTML을 쓰지 않고 React 엘리먼트로 직접 만들어 주입 위험을 없앤다.
import type { ReactNode } from "react";

/** `**굵게**`와 `` `코드` ``만 인라인 처리. 나머지는 그대로 둔다. */
function inline(text: string, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /\*\*(.+?)\*\*|`(.+?)`/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined) out.push(<strong key={`${keyPrefix}-b${i}`}>{m[1]}</strong>);
    else out.push(<code key={`${keyPrefix}-c${i}`}>{m[2]}</code>);
    last = m.index + m[0].length;
    i++;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

type Row = string[];

function splitRow(line: string): Row {
  return line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
}

const isTableSep = (l: string) => /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(l) && l.includes("-");

export default function Markdown({ source, className = "" }: { source: string; className?: string }) {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let para: string[] = [];
  let key = 0;

  const flushPara = () => {
    if (!para.length) return;
    blocks.push(
      <p key={`p${key++}`}>{inline(para.join(" "), `p${key}`)}</p>,
    );
    para = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      flushPara();
      continue;
    }

    // 구분선
    if (/^(-{3,}|_{3,}|\*{3,})$/.test(trimmed)) {
      flushPara();
      blocks.push(<hr key={`hr${key++}`} />);
      continue;
    }

    // 제목
    const h = /^(#{1,4})\s+(.*)$/.exec(trimmed);
    if (h) {
      flushPara();
      const level = h[1].length;
      const content = inline(h[2], `h${key}`);
      const Tag = (["h1", "h2", "h3", "h4"] as const)[level - 1];
      blocks.push(<Tag key={`h${key++}`}>{content}</Tag>);
      continue;
    }

    // 인용 (연속 줄 묶음)
    if (trimmed.startsWith(">")) {
      flushPara();
      const quote: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quote.push(lines[i].trim().replace(/^>\s?/, ""));
        i++;
      }
      i--;
      blocks.push(
        <blockquote key={`q${key++}`}>{inline(quote.join(" "), `q${key}`)}</blockquote>,
      );
      continue;
    }

    // 표 (헤더 + 구분선 + 본문)
    if (trimmed.startsWith("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      flushPara();
      const header = splitRow(trimmed);
      i += 2;
      const rows: Row[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(splitRow(lines[i].trim()));
        i++;
      }
      i--;
      blocks.push(
        <div className="md-table-wrap" key={`t${key++}`}>
          <table>
            <thead>
              <tr>{header.map((c, j) => <th key={j}>{inline(c, `th${j}`)}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri}>{r.map((c, ci) => <td key={ci}>{inline(c, `td${ri}-${ci}`)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    // 목록 · 체크리스트 (연속 줄 묶음)
    if (/^[-*+]\s+/.test(trimmed)) {
      flushPara();
      const items: { text: string; checked: boolean | null }[] = [];
      while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) {
        let body = lines[i].trim().replace(/^[-*+]\s+/, "");
        let checked: boolean | null = null;
        const box = /^\[([ xX])\]\s*/.exec(body);
        if (box) {
          checked = box[1].toLowerCase() === "x";
          body = body.slice(box[0].length);
        }
        // 다음 줄이 들여쓰기된 이어짐이면 합친다
        while (
          i + 1 < lines.length &&
          lines[i + 1].startsWith("  ") &&
          lines[i + 1].trim() &&
          !/^[-*+]\s+/.test(lines[i + 1].trim())
        ) {
          body += " " + lines[i + 1].trim();
          i++;
        }
        items.push({ text: body, checked });
        i++;
      }
      i--;
      const isCheck = items.some((it) => it.checked !== null);
      blocks.push(
        <ul className={isCheck ? "md-checklist" : ""} key={`ul${key++}`}>
          {items.map((it, j) =>
            it.checked === null ? (
              <li key={j}>{inline(it.text, `li${j}`)}</li>
            ) : (
              <li key={j} className={it.checked ? "checked" : ""}>
                <span className="md-box" aria-hidden="true">{it.checked ? "✓" : ""}</span>
                <span>{inline(it.text, `li${j}`)}</span>
              </li>
            ),
          )}
        </ul>,
      );
      continue;
    }

    // 굵은 글씨만 있는 줄은 소제목으로 다룬다.
    // 마크다운 규칙상으로는 다음 줄과 한 문단이 되지만, LLM은 이 형태를
    // 소제목으로 쓰므로 그대로 두면 제목과 본문이 한 덩어리로 붙어 읽힌다.
    const boldOnly = /^\*\*(.+)\*\*$/.exec(trimmed);
    if (boldOnly) {
      flushPara();
      blocks.push(<p className="md-lead" key={`ld${key++}`}>{inline(boldOnly[1], `ld${key}`)}</p>);
      continue;
    }

    para.push(trimmed);
  }
  flushPara();

  return <div className={`md ${className}`.trim()}>{blocks}</div>;
}
