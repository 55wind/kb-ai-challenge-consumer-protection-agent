"""금융위·금감원이 낸 금소법 FAQ·설명자료를 검색 가능한 코퍼스로 만든다.

**왜 이 코퍼스가 따로 필요한가.**

법 조문은 "무엇이 금지되는가"만 말한다. 제21조는 "단정적 판단을 제공하는 행위"를
금지한다고 쓰여 있을 뿐, 소비자가 실제로 쓰는 말("무조건 오른다길래 넣었어요")과
그 조문 사이에는 거리가 있다. 이 거리를 메우려고 감독당국이 직접 낸 것이
FAQ와 설명자료다. 여기에는 조문에 없는 판단 기준이 문장으로 적혀 있다 —
"소액분쟁 여부는 소비자가 신청 시 주장하는 금액으로 판단한다" 같은 것들.

그래서 이 자료는 laws에 섞지 않고 별도 kind로 둔다. 두 가지 이유다.

1. **효력이 다르다.** 조문은 법이고, FAQ는 감독당국의 해석이다. 인용 화면에서
   둘을 같은 무게로 보여주면 사용자가 오해한다. kind가 다르면 출처 표기도 다르게
   할 수 있다.
2. **검색 특성이 다르다.** 조문은 짧고 형식적이라 BM25가 잘 듣고, 해설은 길고
   구어에 가까워 벡터가 잘 듣는다. 한 인덱스에 섞으면 긴 해설 문서가 조문을
   밀어낸다 — laws 골든셋 정확도가 그만큼 깎인다.

**파싱에서 조심한 것.**

FAQ PDF는 문항 번호가 `1.` `2.` 형태인데, 본문에도 `금년9.25일부터`, `3.24일이전에`
같은 숫자가 널려 있다. 정규식만으로 자르면 이런 것들이 전부 가짜 문항이 된다
(실제로 첫 시도에서 11건 중 2건이 가짜였다). 그래서 **번호가 순서대로 이어질 때만**
문항으로 인정한다. 章이 바뀌면 1로 되돌린다. 감독당국 문서는 번호를 건너뛰지
않으므로 이 규칙이 성립한다.
"""
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from pypdf import PdfReader  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "qna_faq"
OUT = Path(__file__).resolve().parent.parent / "data" / "corpus" / "interpretations.jsonl"

# 로마 숫자 章 제목. 문항 번호를 1로 되돌리는 기준점이다.
_SECTION = re.compile(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s*\.\s*[^\d\n]{2,40}?(?=\d{1,2}\.|\n|$)")
# 문항 후보: `<번호>. <제목>□`. □는 감독당국 문서에서 답변 문단의 시작 기호다.
#
# 뒤의 `[ \t]+(?!\d)`가 날짜를 걸러낸다. 본문의 `금년9.25일부터`, `'20.12.1.`은
# 마침표 뒤에 공백이 없거나 숫자가 이어진다. 반대로 앞은 느슨하게 둔다 —
# `…판단합니다.3. 상품숙지의무`처럼 진짜 문항이 마침표에 바로 붙어 나오기 때문에
# 마침표를 막으면 문항을 통째로 잃는다(10문10답에서 10건 중 8건이 이렇게 사라졌다).
#
# 앞의 `[-(]` 제외는 하위 문항을 막는다. 한 문항이 `-1.` `-2.`로 갈라지는 경우가
# 있는데(3차 Ⅱ장 퇴직연금), 이것을 최상위 번호로 세면 번호열이 `1,1,2`가 되어
# 그 章 전체가 순서 규칙에 걸려 버려진다.
_QUESTION = re.compile(r"(?<![-\d(])(\d{1,2})\.[ \t]+(?!\d)([^□\n]{6,250}?)\s*□")
# 문항 묶음 쪼개기. FAQ 3차는 `1. …?2. …?□ <공통 답변>`처럼 두 문항이 답변
# 하나를 나눠 갖는다. 앞 문항만 세면 번호가 어긋나 뒤의 진짜 문항들이 통째로
# 버려진다(3차 Ⅰ장 6문항 중 5문항이 이렇게 사라졌다).
_SUBQ = re.compile(r"(?<![-\d(])(?=\d{1,2}\.[ \t]+(?!\d))")
# 제목 꼬리에 붙는 소절 머리(`가. 위법계약해지권개요`). 답변 본문이 아니라
# 목차 표시라 제목에서 떼어낸다.
_SUBSECTION_TAIL = re.compile(r"[가나다라마바사]\.\s*\S.{1,30}$")
# 쪽 머리(`- 3 -`). 본문 페이지를 가려내는 표식이기도 하다.
_PAGENUM = re.compile(r"^\s*-\s*(\d+)\s*-\s*")
# 조문 인용: `법 제21조`, `금소법 제42조제2호`, `시행령(§16➂)`, `감독규정(§13①5호)`
_CITE_ARTICLE = re.compile(r"제\s*(\d+)\s*조")
_CITE_PARAGRAPH = re.compile(r"[§](\d+)")


@dataclass
class SourceDoc:
    """PDF 하나와 그 출처. source_url은 실제로 그 문서가 게시된 곳이어야 한다."""

    filename: str
    doc_id: str
    source: str
    source_url: str
    kind: str  # "faq" | "해설"


DOCS = [
    SourceDoc(
        "210218_금융소비자보호법 FAQ 답변(1차)_게시용__.pdf",
        "금소법FAQ1차",
        "금융위원회·금융감독원 「금융소비자보호법 FAQ 답변(1차)」 2021.2.18.",
        "https://www.fsc.go.kr/po0201101/75700",
        "FAQ",
    ),
    SourceDoc(
        "210317_금융소비자보호법 FAQ 답변(2차)_게시용__.pdf",
        "금소법FAQ2차",
        "금융위원회·금융감독원 「금융소비자보호법 FAQ 답변(2차)」 2021.3.17.",
        "https://www.fsc.go.kr/po0201101/75701",
        "FAQ",
    ),
    SourceDoc(
        "210331_금소법10문10답_게시용_.pdf",
        "금소법10문10답",
        "금융위원회 「금융소비자보호법 관련 10問 10答」 보도참고자료 2021.3.24.",
        "https://www.fsc.go.kr/po0201101/75663",
        "FAQ",
    ),
    SourceDoc(
        "210426_금융소비자보호법 FAQ 답변(3차)_게시용_.pdf",
        "금소법FAQ3차",
        "금융위원회·금융감독원 「금융소비자보호법 FAQ 답변(3차)」 2021.4.26.",
        # 금융위 정책자료 게시판에는 1·2차만 남아 있다. 3차는 금감원 금소법 FAQ
        # 자료실에 게시돼 있어 그 목록을 출처로 둔다 — 없는 URL을 지어내지 않는다.
        "https://www.fss.or.kr/fss/bbs/B0000112/list.do?menuNo=200140",
        "FAQ",
    ),
    SourceDoc(
        "230919_금융소비자보호법 설명자료_f.pdf",
        "금소법설명자료",
        "금융감독원 「금융소비자 보호에 관한 법률 설명 자료」 2023.9.",
        "https://www.fss.or.kr/fss/bbs/B0000112/list.do?menuNo=200140",
        "해설",
    ),
]


def _pages(path: Path) -> list[str]:
    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def _body_pages(pages: list[str]) -> list[str]:
    """본문만 남긴다.

    표지와 목차에는 쪽 머리(`- 3 -`)가 없다. 감독당국 문서는 예외 없이 본문
    첫 쪽부터 번호를 매기므로, 이 표식 하나로 앞부분을 걷어낼 수 있다.
    """
    out = []
    for t in pages:
        if not _PAGENUM.match(t):
            continue
        out.append(_PAGENUM.sub("", t, count=1))
    return out


def _cited_articles(text: str) -> list[str]:
    """본문이 인용한 조문 번호. 골든셋을 만들 때 정답 근거로 쓴다."""
    nums = {m.group(1) for m in _CITE_ARTICLE.finditer(text)}
    nums |= {m.group(1) for m in _CITE_PARAGRAPH.finditer(text)}
    return sorted((f"제{n}조" for n in nums), key=lambda s: int(s[1:-1]))


def parse_faq(doc: SourceDoc) -> list[dict]:
    """번호가 순서대로 이어지는 것만 문항으로 인정한다 (모듈 설명 참조)."""
    full = "\n".join(_body_pages(_pages(RAW_DIR / doc.filename)))
    section_at = [(m.start(), m.group(0).strip()) for m in _SECTION.finditer(full)]

    accepted: list[tuple[re.Match, str, str]] = []
    expected = 1
    topic = ""
    for m in _QUESTION.finditer(full):
        # 이 문항 앞에 새 章이 있으면 번호를 1로 되돌린다.
        crossed = [s for s in section_at if s[0] < m.start()]
        if crossed and crossed[-1][1] != topic:
            topic = crossed[-1][1]
            expected = 1
        # 한 매치 안에 문항이 여럿일 수 있다 — 번호가 expected부터 연속인지 본다.
        parts = [p.strip() for p in _SUBQ.split(f"{m.group(1)}. {m.group(2)}") if p.strip()]
        nums, texts = [], []
        for p in parts:
            head, _, rest = p.partition(".")
            if not head.isdigit():
                # 앞 문항 제목이 이어진 조각. 직전 문항에 되붙인다.
                if texts:
                    texts[-1] += " " + p
                continue
            nums.append(int(head))
            texts.append(rest.strip())
        if nums != list(range(expected, expected + len(nums))):
            continue  # `금년9.25일부터` 같은 가짜 번호이거나 순서가 어긋났다
        title = " / ".join(_clean_title(t) for t in texts)
        accepted.append((m, topic, title))
        expected = nums[-1] + 1

    records = []
    for i, (m, topic, title) in enumerate(accepted):
        end = accepted[i + 1][0].start() if i + 1 < len(accepted) else len(full)
        # m.end()는 답변 첫 □를 이미 지났다 — 그 □부터 답변으로 삼는다.
        answer = full[m.end() - 1:end].strip()
        # 다음 章 제목이 답변 꼬리에 붙는 경우를 잘라낸다.
        answer = _SECTION.sub("", answer).strip()
        if len(answer) < 40:
            continue  # 본문이 사실상 비었으면 근거로 못 쓴다
        records.append({
            "id": f"{doc.doc_id}-{i + 1}",
            "source": doc.source,
            "source_url": doc.source_url,
            "doc_type": doc.kind,
            "topic": topic or doc.doc_id,
            "title": title,
            "text": answer,
            "cited_articles": _cited_articles(answer),
        })
    return records


def _clean_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", raw).strip()
    stripped = _SUBSECTION_TAIL.sub("", title).strip()
    # 소절 머리를 떼고 나서도 제목 구실을 할 만큼 남아야 뗀다.
    return stripped if len(stripped) >= 6 else title


def parse_guide(doc: SourceDoc) -> list[dict]:
    """해설서는 문답 구조가 없다 — 쪽 단위로 자르고 직전 章 제목을 붙인다.

    쪽 단위인 이유: 이 문서는 한 쪽이 한 논점으로 떨어지게 편집돼 있다
    (예: 39쪽 전체가 '부당권유행위 금지 위반의 효과'). 문장 단위로 더 쪼개면
    "위반 시 과징금은 수입의 50% 이내" 같은 문장이 무슨 위반에 대한 것인지
    잃어버린다.
    """
    body = _body_pages(_pages(RAW_DIR / doc.filename))
    records = []
    topic = ""
    for i, text in enumerate(body, 1):
        secs = _SECTION.findall(text)
        if secs:
            topic = secs[0].strip()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 200:
            continue  # 도표만 있는 쪽 — 텍스트가 없으면 검색에도 안 걸린다
        # 제목은 쪽 첫 문장에서 딴다. 없으면 章 제목으로 대신한다.
        head = re.split(r"[□◦※]", text, maxsplit=1)[0].strip()
        title = head[:60] if len(head) >= 6 else (topic or f"{i}쪽")
        records.append({
            "id": f"{doc.doc_id}-p{i}",
            "source": doc.source,
            "source_url": doc.source_url,
            "doc_type": doc.kind,
            "topic": topic or doc.doc_id,
            "title": title,
            "text": text,
            "cited_articles": _cited_articles(text),
        })
    return records


def main() -> int:
    all_records: list[dict] = []
    for doc in DOCS:
        path = RAW_DIR / doc.filename
        if not path.exists():
            print(f"  [건너뜀] 원본 없음: {doc.filename}")
            continue
        recs = parse_faq(doc) if doc.kind == "FAQ" else parse_guide(doc)
        print(f"  {doc.doc_id}: {len(recs)}건")
        all_records.extend(recs)

    ids = [r["id"] for r in all_records]
    assert len(ids) == len(set(ids)), "id가 중복됐다 — 인덱스와 코퍼스가 어긋난다"

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_records) + "\n",
        encoding="utf-8",
    )
    chars = sum(len(r["text"]) for r in all_records)
    print(f"\n총 {len(all_records)}건 · 본문 {chars:,}자 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
