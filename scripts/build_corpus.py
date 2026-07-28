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


def _category(title: str, body: str) -> str:
    for cat, kws in _CATEGORY_RULES:
        if any(k in title for k in kws):
            return cat
    for cat, kws in _CATEGORY_RULES:
        if any(k in body[:2000] for k in kws):
            return cat
    return "기타"


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


def parse_case(text: str, source_url: str, title_hint: str, file_name: str) -> CaseRecord | None:
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
    return CaseRecord.validated(
        id=f"fss-{case_no}-{file_name.rsplit('.', 1)[0]}",
        title=title,
        facts=facts[:4000],
        issues=_section(text, line_chwiji, [line_iyu])[:1500] or title,
        reasoning=judging[:5000],
        decision=_decision(judgment),
        decision_date=decision_date,
        category=_category(title, text),
        source=f"금융감독원 금융분쟁조정위원회 조정결정서 {case_no}",
        source_url=source_url,
    )


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
            text = extract_hwp_text(path)
            rec = parse_case(
                text,
                source_url=meta.get("page_url", "https://www.fss.or.kr/fss/bbs/B0000390/list.do"),
                title_hint=meta.get("title", path.stem),
                file_name=path.name,
            )
        except (HwpParseError, Exception) as e:  # noqa: BLE001 - 수집물 품질 편차 흡수
            skipped.append(f"{path.name}: {e}")
            continue
        if rec is None:
            skipped.append(f"{path.name}: 섹션 파싱 실패")
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
                m = re.match(r"제\s*(\d+)\s*조", part)
                if m and len(part) >= 40:
                    chunks.append((f"제{m.group(1)}조", part[:3000]))
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
