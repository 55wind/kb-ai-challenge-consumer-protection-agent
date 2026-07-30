"""raw 수집물 → 구조화 JSONL 코퍼스.

usage:
  python scripts/build_corpus.py cases       # data/raw/cases/*.hwp → cases.jsonl
  python scripts/build_corpus.py laws        # data/raw/laws/*.txt → laws.jsonl
  python scripts/build_corpus.py procedures  # data/raw/procedures/*.json → procedures.jsonl

파싱 실패 파일은 스킵하고 목록을 출력한다(수동 보정 대상).
사례 결정(인용/일부인용/기각)은 주문 텍스트 휴리스틱으로 라벨링한다.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.corpus.hwp import HwpParseError, extract_hwp_text
from app.corpus.schema import CaseRecord, LawChunk, ProcedureDoc, ProductChunk

RAW = settings.DATA_DIR / "raw"

_CATEGORY_RULES = [
    ("보이스피싱·전자금융사기", ["보이스피싱", "전자금융", "명의도용", "피싱", "부정사용", "해킹"]),
    ("불완전판매", ["불완전판매", "사모펀드", "펀드", "신탁", "ELS", "설명의무", "투자", "적합성"]),
    ("대출·금리", ["대출", "금리", "이자", "연체", "근저당", "상환", "한도"]),
    ("카드", ["신용카드", "카드", "할부", "청약철회"]),
    ("보험", ["보험", "상해", "후유장해", "교통사고", "진단", "입원", "보험금"]),
]


def _norm_title(s: str) -> str:
    """제목 대조용 정규화. 사례집 PDF는 추출 시 공백이 사라져 양쪽에서 다 지운다."""
    return re.sub(r"[\s·․,()（）:：'\"‘’“”\-~/]", "", s)


_FSS_INDEX: dict[str, str] | None = None


def _fss_category(title: str) -> str | None:
    """금융감독원 「금융분쟁조정 사례집」이 이 사건에 붙인 분류.

    사례집 목차에는 조정사례 410건에 금감원이 직접 붙인 3단계 분류가 있다
    (`scripts/build_fss_case_index.py`). 제목이 같으면 그 분류를 쓴다 —
    우리 키워드 규칙보다 이쪽이 옳다.
    """
    global _FSS_INDEX
    if _FSS_INDEX is None:
        from scripts.build_classifier_goldset import CHAPTER_MAP

        path = settings.DATA_DIR / "fss_case_index.jsonl"
        _FSS_INDEX = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                cat = CHAPTER_MAP.get(e["chapter"])
                if cat:
                    _FSS_INDEX[_norm_title(e["title"])] = cat
    n = _norm_title(title)
    if n in _FSS_INDEX:
        return _FSS_INDEX[n]
    # 사례집 제목이 더 길거나 짧게 실린 경우가 있다 — 충분히 길 때만 포함으로 본다.
    if len(n) > 15:
        for k, v in _FSS_INDEX.items():
            if n in k or k in n:
                return v
    return None


def _category(title: str, body: str) -> tuple[str, str]:
    """(유형, 그 유형을 누가 정했는지).

    금감원 사례집에 있으면 그쪽을 쓴다. 없을 때만 키워드 규칙으로 추정한다.
    추정은 틀린다 — 금감원 라벨과 대조 가능한 17건에서 3건(18%)이 어긋났다.
    `계약전 발병 부담보 조항의 유효성 여부`가 '대출·금리'로, 증권 임의매매 사건이
    '보이스피싱·전자금융사기'로 잡혔다. 그래서 출처를 레코드에 함께 남긴다.
    """
    if fss := _fss_category(title):
        return fss, "금융감독원 사례집"
    for cat, kws in _CATEGORY_RULES:
        if any(k in title for k in kws):
            return cat, "키워드 규칙(추정)"
    for cat, kws in _CATEGORY_RULES:
        if any(k in body[:2000] for k in kws):
            return cat, "키워드 규칙(추정)"
    return "기타", "키워드 규칙(추정)"


def _decision(judgment: str) -> str:
    has_reject = "기각한다" in judgment or "기각 한다" in judgment
    has_grant = bool(re.search(r"(지급한다|지급하라|배상한다|배상하라|이행한다|이행하라|취소한다)", judgment))
    if has_reject and has_grant:
        return "일부인용"
    if has_reject:
        return "기각"
    if has_grant:
        return "인용"
    return "기각"


def _section(text: str, start_pat: str, end_pats: list[str]) -> str:
    m = re.search(start_pat, text)
    if not m:
        return ""
    rest = text[m.end():]
    end = len(rest)
    for pat in end_pats:
        em = re.search(pat, rest)
        if em and em.start() < end:
            end = em.start()
    return rest[:end].strip()


# 조정결정서의 제목 표시는 **넓은 자간**이다 — `주       문`, `이       유`.
# 본문에서 '이유'를 쓸 때는 붙여 쓰므로 이 간격이 제목을 가르는 신호가 된다.
_HEADING = re.compile(r"(주[ \t]{2,}문|이[ \t]{2,}유|신[ \t]*청[ \t]*취[ \t]*지)")
# 구형 서식은 제목에 번호를 붙인다 — `3. 주    문`, `5. 이   유`.
_NUMBERED_HEADING = re.compile(r"(?m)^[ \t]*\d+\.[ \t]*(?=주[ \t]*문|이[ \t]*유|신[ \t]*청[ \t]*취[ \t]*지)")


def _normalize_sections(text: str) -> str:
    """제목을 각자 한 줄에 놓는다.

    같은 결정서인데 첨부 형식에 따라 제목이 줄 안에 파묻힌다.

      - PDF로 올라온 결정서는 추출 시 줄바꿈이 사라져
        `피 신 청 인 ㈜Y보험 주       문피신청인은…`처럼 한 줄이 된다.
      - 2016년 이전 서식은 `3. 주    문`처럼 번호가 붙는다.

    섹션 정규식을 줄 시작에 묶어 둔 것은 본문 속 '주문'·'이유'에 걸리지 않기
    위해서다. 그 안전장치를 풀지 않고, 대신 **제목을 줄로 떼어 준다.**
    """
    text = _NUMBERED_HEADING.sub("", text)
    return _HEADING.sub(lambda m: f"\n{m.group(1)}\n", text)


def parse_case(text: str, source_url: str, title_hint: str, file_name: str) -> CaseRecord | None:
    text = _normalize_sections(text)
    num_m = re.search(r"조정번호\s*[:：]\s*(제?\s*[\d]{4}\s*-\s*\d+\s*호)", text)
    date_m = re.search(r"조정일자\s*[:：]\s*([\d]{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", text)
    title_m = re.search(r"안\s*건\s*명\s+(.+)", text)
    line_ju = r"(?m)^\s*주[ \t]{0,20}문[ \t]*$"
    line_chwiji = r"(?m)^\s*신[ \t]*청[ \t]*취[ \t]*지[ \t]*$"
    line_iyu = r"(?m)^\s*이[ \t]{0,20}유[ \t]*$"
    judgment = _section(text, line_ju, [line_chwiji, line_iyu])
    reason = _section(text, line_iyu, [r"(?m)^\s*별\s*지\s*1", r"이상과 같은 이유로"])
    if not judgment or not reason:
        return None
    facts = _section(reason, r"기\s*초\s*사\s*실", [r"당\s*사\s*자\s*(의)?\s*주\s*장", r"위\s*원\s*회\s*(의)?\s*판\s*단"]) or reason[:1500]
    judging = _section(reason, r"위\s*원\s*회\s*(의)?\s*판\s*단", [r"\Z"]) or reason[-3000:]
    title = (title_m.group(1).strip() if title_m else title_hint).strip()
    case_no = re.sub(r"\s", "", num_m.group(1)) if num_m else file_name
    decision_date = f"{date_m.group(1)}-{int(date_m.group(2)):02d}-{int(date_m.group(3)):02d}" if date_m else ""
    category, category_source = _category(title, text)
    return CaseRecord.validated(
        id=f"fss-{case_no}-{file_name.rsplit('.', 1)[0]}",
        title=title,
        facts=facts[:4000],
        issues=_section(text, line_chwiji, [line_iyu])[:1500] or title,
        reasoning=judging[:5000],
        decision=_decision(judgment),
        decision_date=decision_date,
        category=category,
        category_source=category_source,
        source=f"금융감독원 금융분쟁조정위원회 조정결정서 {case_no}",
        source_url=source_url,
    )


def _case_texts(path: Path) -> list[tuple[str, str]]:
    """결정서 하나에서 (파일이름, 본문) 목록. 첨부가 여러 형식으로 온다.

    분쟁조정결정례 게시판의 첨부는 확장자가 전부 `.hwp`인데 실제 내용은 셋이다.
    이걸 확장자로만 믿었더니 **34건이 미분석으로 빠져 있었다.**

      - OLE(진짜 HWP)  : 지금까지 읽던 것
      - ZIP            : 결정서 여러 건을 묶은 것. 2019~2021년 안건 12건이
                         이 안에 들어 있었다(`의결안건 제2020-11호(게시용).hwp` …)
      - PDF            : 결정서를 PDF로 올린 것, 그리고 사례집 두 권
    """
    head = path.open("rb").read(8)

    if head[:4] == b"%PDF":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        # 사례집(수백 쪽)은 결정서가 아니다 — 유형 색인으로만 쓴다
        # (`scripts/build_fss_case_index.py`).
        if len(reader.pages) > 60:
            return []
        return [(path.name, "\n".join((p.extract_text() or "") for p in reader.pages))]

    if head[:2] == b"PK":
        import tempfile
        import zipfile

        out = []
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith((".hwp", ".hwpx")):
                    continue
                try:
                    name = info.filename.encode("cp437").decode("cp949")
                except Exception:  # noqa: BLE001 - 이미 유니코드인 경우
                    name = info.filename
                with tempfile.NamedTemporaryFile(suffix=".hwp", delete=False) as tmp:
                    tmp.write(z.read(info))
                    tmp_path = Path(tmp.name)
                try:
                    out.append((f"{path.stem}::{Path(name).name}", extract_hwp_text(tmp_path)))
                except Exception:  # noqa: BLE001 - 묶음 안의 한 건이 깨져도 나머지는 살린다
                    pass
                finally:
                    tmp_path.unlink(missing_ok=True)
        return out

    return [(path.name, extract_hwp_text(path))]


def build_cases() -> None:
    raw_dir = RAW / "cases"
    manifest = {}
    mpath = raw_dir / "manifest.tsv"
    if mpath.exists():
        with mpath.open(encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                manifest[row["file"]] = row
    records, skipped, seen_ids = [], [], set()
    for path in sorted(raw_dir.glob("*.hwp")):
        meta = manifest.get(path.name, {})
        try:
            parts = _case_texts(path)
        except Exception as e:  # noqa: BLE001 - 수집물 품질 편차 흡수
            skipped.append(f"{path.name}: {e}")
            continue
        if not parts:
            skipped.append(f"{path.name}: 읽을 수 있는 본문 없음")
            continue
        for member, text in parts:
            try:
                rec = parse_case(
                    text,
                    source_url=meta.get("page_url",
                                        "https://www.fss.or.kr/fss/bbs/B0000390/list.do"),
                    title_hint=meta.get("title", path.stem),
                    file_name=member,
                )
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{member}: {e}")
                continue
            if rec is None:
                skipped.append(f"{member}: 섹션 파싱 실패")
                continue
            if rec.id in seen_ids:
                continue
            seen_ids.add(rec.id)
            records.append(rec)
    settings.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out = settings.CORPUS_DIR / "cases.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
    print(f"cases: {len(records)}건 기록, {len(skipped)}건 스킵 → {out}")
    for s in skipped:
        print(f"  [skip] {s}")


def build_laws() -> None:
    raw_dir = RAW / "laws"
    records = []
    for path in sorted(raw_dir.glob("*.txt")):
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        law_name = meta.get("law_name", path.stem)
        source_url = meta.get("source_url", "")
        text = path.read_text(encoding="utf-8")
        # 조문 단위 분리: "제N조(제목)" 또는 "제N조의M(제목)"
        # 부칙은 조 번호가 제1조로 리셋된다(단조성 파괴) → 그 지점부터 중단해
        # 본문 조문과의 ID 충돌·검색 오염을 방지한다.
        parts = re.split(r"\n(?=제\s*\d+\s*조(?:의\s*\d+)?\s*\()", text)
        prev_no = 0
        seen_ids: set[str] = set()
        for part in parts:
            part = part.strip()
            m = re.match(r"제\s*(\d+)\s*조(?:의\s*(\d+))?\s*\(([^)]+)\)", part)
            if not m or len(part) < 30:
                continue
            article_no = int(m.group(1))
            if article_no < prev_no:
                break  # 부칙 시작 (조 번호 리셋)
            prev_no = article_no
            suffix = f"의{m.group(2)}" if m.group(2) else ""
            article = f"제{article_no}조{suffix}"
            # 현행/시행예정 교차 표기 등으로 같은 조가 반복되면 첫 등장(현행)만 유지
            if f"{law_name}-{article}" in seen_ids:
                continue
            seen_ids.add(f"{law_name}-{article}")
            records.append(
                LawChunk.validated(
                    id=f"{law_name}-{article}",
                    law_name=law_name,
                    article=f"{article}({m.group(3)})",
                    text=part[:4000],
                    source=meta.get("source", f"국가법령정보센터 {law_name}"),
                    source_url=source_url,
                )
            )
    out = settings.CORPUS_DIR / "laws.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
    print(f"laws: {len(records)}건 기록 → {out}")


def build_procedures() -> None:
    raw_dir = RAW / "procedures"
    records = []
    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append(ProcedureDoc.validated(**data))
    out = settings.CORPUS_DIR / "procedures.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
    print(f"procedures: {len(records)}건 기록 → {out}")


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def build_kb_products() -> None:
    """KB 약관·상품설명서 PDF → 조항(약관) 또는 단락(설명서) 청크."""
    raw_dir = RAW / "kb_products"
    records = []
    for path in sorted(raw_dir.glob("*.pdf")):
        name = path.stem
        doc_type = "약관" if "약관" in name else ("특약" if "특약" in name else "상품설명서")
        product_name = name.replace("_", " ")
        text = _pdf_text(path)
        if doc_type in ("약관", "특약"):
            parts = re.split(r"(?=제\s*\d+\s*조(?:의\s*\d+)?\s*[(\[])", text)
            chunks = []
            for part in parts:
                part = part.strip()
                # `의2` 가지번호까지 잡아야 한다. 분할 정규식은 `제9조의2`를
                # 알아보는데 라벨에서 `의2`를 버리면 제9조와 제9조의2가 같은
                # id를 갖는다. 그러면 `_get_records`가 dict라 **하나가 다른
                # 하나를 덮어쓴다** — 실제로 예금거래기본약관 제9조(이자)가
                # 검색에서 사라지고, 그 벡터에 맞아도 제9조의2(휴면예금) 본문이
                # 인용문으로 나갔다. 근거와 인용이 어긋나는 상태였다.
                m = re.match(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?", part)
                if m and len(part) >= 40:
                    section = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
                    chunks.append((section, part[:3000]))
        else:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 80]
            body = "\n".join(paragraphs) if paragraphs else text
            chunks = [
                (f"p{i + 1}", body[i * 900 : i * 900 + 1000])
                for i in range(min(12, max(1, len(body) // 900)))
                if body[i * 900 : i * 900 + 1000].strip()
            ]
        for section, chunk_text in chunks:
            records.append(ProductChunk.validated(
                id=f"kb-{name}-{section}",
                product_name=product_name,
                doc_type=doc_type,
                section=section,
                text=chunk_text,
                source=f"KB국민은행 {product_name} ({doc_type})",
                source_url="https://obank.kbstar.com/quics?page=C022180",
            ))
    # id가 겹치면 검색은 되는데 조회에서 하나가 사라진다(`_get_records`가 dict다).
    # 조용히 지나가면 인용문이 검색된 문서와 달라지므로 여기서 멈춘다.
    dup = [i for i, c in Counter(r.id for r in records).items() if c > 1]
    if dup:
        raise SystemExit(f"kb_products id 중복: {dup} — 조항 라벨 규칙을 확인하세요")

    settings.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out = settings.CORPUS_DIR / "kb_products.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
    print(f"kb_products: {len(records)}건 기록 → {out}")


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "cases"
    {
        "cases": build_cases,
        "laws": build_laws,
        "procedures": build_procedures,
        "kb_products": build_kb_products,
    }[kind]()
