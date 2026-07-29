"""금융감독원 「금융분쟁조정 사례집」의 **사례 목차**를 구조화한다.

**왜 필요한가 — 지금 정답지가 우리가 만든 것이기 때문이다.**

`cases.jsonl`의 `category`는 금감원이 붙인 값이 아니라 `build_corpus.py`의
`_category(title, body)` — **우리가 짠 키워드 규칙**이 만든 값이다. 그런데
문서들은 그것을 "기관이 분류한 실데이터"라고 적어 왔다. 사실이 아니다.

진짜 라벨은 원본에 있었다. 금감원이 2013년에 낸 「금융분쟁조정 사례집」
두 권(보험 / 은행·중소서민금융·금융투자)의 **사례 목차**는 조정사례 하나하나에
금감원이 직접 3단계 분류를 붙여 놓은 것이다.

    제1편 주제별 주요 조정사례(Ⅰ)
      제1장 계약전 알릴의무 위반 및 계약해지 등
        제1절 암 관련
          1. 만성위염 치료사실 미고지로 인한 계약해지의 적정성 여부(2000) … 3

이 목차를 뽑아 `제목 → (편, 장, 절, 연도)` 색인을 만든다. 우리 코퍼스의 결정례와
**제목으로 이어 붙이면** 우리가 지어낸 라벨을 금감원 라벨로 바꿀 수 있다.

**출처.** 두 권 모두 `data/raw/cases/`에 이미 있었다. 분쟁조정결정례 게시판에서
결정서와 함께 내려받힌 첨부물인데, 확장자가 `.hwp`인데 실제로는 PDF라
HWP 파서가 열지 못해 **34건의 미분석 파일 속에 묻혀 있었다.**

  - `case_64766_2.hwp` → 실제로는 PDF. 사례집 보험편 (964쪽)
  - `case_64767_2.hwp` → 실제로는 PDF. 사례집 은행·중소서민금융·금융투자편 (798쪽)

**파싱에서 조심한 것.** 보험편 PDF는 텍스트 추출 시 공백이 사라진다
("1.만성위염치료사실미고지로…"). 그래서 제목 대조는 **양쪽에서 공백을 모두
제거하고** 한다. 은행편은 공백이 살아 있어 같은 규칙이 그대로 통한다.
"""
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from pypdf import PdfReader  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "cases"
OUT = ROOT / "data" / "fss_case_index.jsonl"

# (파일, 권 이름, 출처 표기)
VOLUMES = [
    ("case_64766_2.hwp", "보험",
     "금융감독원 「금융분쟁조정 사례집(제1권) — 보험」 2013.12."),
    ("case_64767_2.hwp", "은행·중소서민금융·금융투자",
     "금융감독원 「금융분쟁조정 사례집(제1권) — 은행, 중소서민금융, 금융투자」 2013.12."),
]
SOURCE_URL = "https://www.fss.or.kr/fss/bbs/B0000390/list.do"

# 목차 표제. 공백이 사라진 PDF와 살아 있는 PDF 양쪽에서 잡히게 \s*를 넉넉히 둔다.
_PART = re.compile(r"제\s*(\d)\s*편\s*([^\n제]{2,40}?)(?=제\s*\d\s*장|\n|$)")
_CHAPTER = re.compile(r"제\s*(\d{1,2})\s*장\s*([^\n]{2,40}?)(?=제\s*\d{1,2}\s*절|\n|$)")
_SECTION = re.compile(r"제\s*(\d{1,2})\s*절\s*([^…\n]{2,40}?)\s*…")
# 사례 한 줄: `12. 제목(2008) … 45`
#
# 번호 앞에 절 제목의 쪽번호가 붙어 나온다 — `제1절 암 관련 … 3` 다음에 `1.`이
# 오면 텍스트에서는 `…31.만성위염…`이 된다. 31번인지 3쪽+1번인지 정규식만으로는
# 가를 수 없다. 그래서 숫자 뭉치를 통째로 잡아 두고, **장 안에서 번호가 1부터
# 순서대로 이어진다**는 성질로 가른다(FAQ 파서와 같은 방식).
#
# 쪽번호는 **전방탐색으로만** 본다. 소비해 버리면 그 숫자 뭉치에 다음 사례 번호가
# 붙어 있을 때 그것까지 먹는다 — `… 37` 다음에 `10.`이 오면 `3710`을 쪽번호로
# 삼고 10번 사례를 통째로 잃는다(실제로 장마다 9건에서 끊겼다).
_ENTRY = re.compile(
    r"(\d{1,4})\s*\.\s*([^…]{8,200}?)\s*[（(](\d{4})[)）](?=\s*…\s*(\d{1,4}))")


@dataclass
class Entry:
    volume: str
    part: str
    chapter: str
    section: str
    no: int
    title: str
    year: int
    page: int
    source: str
    source_url: str


def _toc_text(path: Path) -> str:
    """'사례목차'부터 부록 직전까지. 앞의 요약 목차는 사례 단위가 아니라 건너뛴다."""
    reader = PdfReader(str(path))
    # 사례 목차는 로마자 쪽번호 구간(앞부분)에 있다. 넉넉히 40쪽을 읽는다.
    text = "\n".join((p.extract_text() or "") for p in reader.pages[:40])
    text = re.sub(r"[·．.]{2,}", " … ", text)
    start = text.find("사례목차")
    if start < 0:
        start = text.find("사례 목차")
    return text[start:] if start >= 0 else text


def parse_volume(path: Path, volume: str, source: str) -> list[Entry]:
    text = _toc_text(path)
    out: list[Entry] = []
    part = chapter = section = ""
    # 표제와 항목이 한 줄에 섞여 나오므로, 위치 순서대로 훑으며 문맥을 갱신한다.
    marks = []
    for m in _PART.finditer(text):
        marks.append((m.start(), "part", m.group(2).strip()))
    for m in _CHAPTER.finditer(text):
        marks.append((m.start(), "chapter", m.group(2).strip()))
    for m in _SECTION.finditer(text):
        marks.append((m.start(), "section", m.group(2).strip()))
    for m in _ENTRY.finditer(text):
        marks.append((m.start(), "entry", m))
    marks.sort(key=lambda x: x[0])

    expected = 1  # 장이 바뀔 때 1로 되돌린다
    for _, kind, val in marks:
        if kind == "part":
            part, chapter, section = val, "", ""
            expected = 1
        elif kind == "chapter":
            chapter, section = val, ""
            expected = 1
        elif kind == "section":
            section = val
        else:
            digits = val.group(1)
            # 숫자 뭉치의 꼬리가 기대 번호와 같아야 한다. 앞자리는 절 제목의 쪽번호다.
            if not digits.endswith(str(expected)):
                continue
            title = re.sub(r"\s+", "", val.group(2))
            if len(title) < 8:
                continue
            out.append(Entry(
                volume=volume, part=part, chapter=chapter, section=section,
                no=expected, title=title, year=int(val.group(3)),
                page=int(val.group(4)), source=source, source_url=SOURCE_URL,
            ))
            expected += 1
    return out


def main() -> int:
    all_entries: list[Entry] = []
    for fname, volume, source in VOLUMES:
        path = RAW / fname
        if not path.exists():
            print(f"  [건너뜀] 원본 없음: {fname}")
            continue
        entries = parse_volume(path, volume, source)
        chapters = {e.chapter for e in entries if e.chapter}
        print(f"  {volume}: 사례 {len(entries)}건 · 장 {len(chapters)}종")
        all_entries.extend(entries)

    OUT.write_text(
        "\n".join(json.dumps(asdict(e), ensure_ascii=False) for e in all_entries) + "\n",
        encoding="utf-8")
    print(f"\n총 {len(all_entries)}건 → {OUT}")

    print("\n금감원 분류 체계 (장 단위)")
    seen = []
    for e in all_entries:
        key = (e.volume, e.part, e.chapter)
        if e.chapter and key not in seen:
            seen.append(key)
    for vol, part, ch in seen:
        n = sum(1 for e in all_entries if (e.volume, e.part, e.chapter) == (vol, part, ch))
        print(f"  [{vol}] {part} / {ch}  — {n}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
