"""금융위원회 법령해석 회신사례를 수집한다 — RAG 근거 보강용.

## 무엇을 왜 가져오나

`data/corpus/interpretations.jsonl`은 금융위 「금소법 FAQ」 130건이다. 조문(무엇이
규정인가)과 결정례(어떻게 판단됐나)는 있는데, **조문의 뜻이 갈리는 자리에서 감독당국이
무엇이라 답했는가**는 FAQ가 다루는 범위까지만 있다.

금융규제·법령해석포털(better.fsc.go.kr)의 법령해석 회신사례는 그 빈칸을 메운다.
구조가 그대로 근거로 쓰기 좋다 — 질의요지 / 회답 / 이유(조문 인용).

## 저작권

금융위원회는 중앙행정기관이므로 **저작권법 제24조의2(공공저작물의 자유이용)** ①항이
직접 적용된다 — "국가 또는 지방자치단체가 업무상 작성하여 공표한 저작물은 허락 없이
이용할 수 있다." 같은 항의 예외 4개(국가안전보장 / 사생활·사업상 비밀 / 다른 법률상
공개제한 / 국유재산 등록)에 해당하지 않는다.

사이트 「저작권정책」 페이지에는 "무단 복제 및 배포를 원칙적으로 금합니다"라고
적혀 있으나, 이는 제24조의2 신설(2014) 이전의 문구가 남은 것으로 보인다(같은 페이지
푸터가 `COPYRIGHT 2009`, 공공누리 표시 없음). 법률 규정이 안내문에 우선한다.
그래도 레코드마다 출처와 원문 URL을 남기고, 본문을 고치지 않는다.

## 선별 기준을 우리가 정하지 않는다

키워드로 후보를 모으면 "무엇이 소비자보호 관련인가"를 우리가 판단하게 된다. 그래서
**소관부서가 금융소비자정책과인 것만** 남긴다 — 금융위 자신의 편철이다. 사례집 라벨을
금감원 章에서 가져온 것과 같은 원리다.

수집: python scripts/collect_fsc_interpretations.py            (원문 → data/raw/)
반영: python scripts/collect_fsc_interpretations.py --merge    (코퍼스에 병합)
"""
import argparse
import io
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "fsc_interpretations"
CORPUS = ROOT / "data" / "corpus" / "interpretations.jsonl"

BASE = "https://better.fsc.go.kr/fsc_new"
LIST_URL = f"{BASE}/replyCase/selectReplyCasePastReplyList.do"
LIST_PAGE = f"{BASE}/replyCase/PastReplyList.do?stNo=11&muNo=171&muGpNo=75"
LAW_DETAIL = f"{BASE}/replyCase/LawreqDetail.do"
OPINION_DETAIL = f"{BASE}/replyCase/OpinionDetail.do"

# 후보 수집용 검색어. 이것으로 최종 선별을 하지 않는다 — 그물을 넓게 던지고,
# 남길지는 소관부서가 정한다.
KEYWORDS = [
    "금융소비자보호", "설명의무", "적합성", "적정성", "불공정영업", "부당권유",
    "위법계약해지", "청약철회", "투자광고", "금융상품 광고", "보이스피싱",
    "착오송금", "예금자보호", "약관", "금융상품판매", "손해배상",
]
# 금융위 자신의 편철. 이 부서가 소관인 회신만 남긴다.
KEEP_DEPT = "금융소비자정책과"

TAG = re.compile(r"<[^>]+>")
DEPT = re.compile(r"소관부서\s+(\S{2,40}?(?:과|팀|국|실))\s")
DELAY = 0.5   # 공공 사이트에 부담을 주지 않는다


@dataclass
class Reply:
    id: str
    source: str
    source_url: str
    doc_type: str
    topic: str
    title: str
    text: str
    cited_articles: list[str]
    reply_date: str
    department: str


def clean(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(p|div|li|tr|h\d)>", "\n", html)
    text = TAG.sub("", html).replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"[ \t]+", " ", text)


def section(text: str, start: str, ends: tuple[str, ...]) -> str:
    """`start` 다음부터 `ends` 중 가장 먼저 오는 것 전까지.

    표 범례에도 '질의요지'가 나오므로 **마지막** 출현을 기준으로 한다 — 범례는
    본문보다 앞에 있다.
    """
    i = text.rfind(start)
    if i < 0:
        return ""
    i += len(start)
    stops = [p for p in (text.find(e, i) for e in ends) if p > 0]
    return text[i: min(stops) if stops else len(text)].strip()


_LAW_ALIAS = {
    "금융소비자보호에관한법률시행령": "금융소비자 보호에 관한 법률 시행령",
    "금융소비자보호법시행령": "금융소비자 보호에 관한 법률 시행령",
    "금융소비자보호에관한감독규정": "금융소비자 보호에 관한 감독규정",
    "금융소비자보호감독규정": "금융소비자 보호에 관한 감독규정",
    "금융소비자보호에관한법률": "금융소비자 보호에 관한 법률",
    "금융소비자보호법": "금융소비자 보호에 관한 법률",
    "금소법": "금융소비자 보호에 관한 법률",
}
_CITE = re.compile(
    r"(?P<law>금융소비자\s*보호에\s*관한\s*법률\s*시행령|금융소비자\s*보호에\s*관한\s*감독규정"
    r"|금융소비자\s*보호\s*감독규정|금융소비자\s*보호에\s*관한\s*법률"
    r"|금융소비자보호법\s*시행령|금융소비자보호법|금소법)"
    r"[^가-힣0-9]{0,14}(?:제\s*)?(?P<art>\d{1,3})\s*조"
    r"|(?P<law2>금융소비자보호법|금소법|법)\s*[(（]?\s*§\s*(?P<art2>\d{1,3})"
)


def cited(text: str) -> list[str]:
    """본문이 인용한 금소법 계열 조문. 코퍼스 id 형식(`법명-제N조`)으로 돌려준다.

    다른 법(보험업법·자본시장법 등)은 코퍼스에 없어 연결할 수 없으므로 담지 않는다.
    담으면 화면이 있지도 않은 근거를 가리킨다.
    """
    out = []
    for m in _CITE.finditer(text):
        raw = (m.group("law") or m.group("law2") or "").replace(" ", "")
        art = m.group("art") or m.group("art2")
        canon = _LAW_ALIAS.get(raw)
        if canon and art:
            cid = f"{canon}-제{art}조"
            if cid not in out:
                out.append(cid)
    return out


def make_client() -> httpx.Client:
    c = httpx.Client(
        timeout=60, follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest",
                 "Referer": LIST_PAGE},
    )
    c.get(LIST_PAGE)          # 세션 쿠키. 없으면 목록 API가 빈 결과를 준다
    return c


def collect_candidates(c: httpx.Client) -> dict[int, dict]:
    seen: dict[int, dict] = {}
    for kw in KEYWORDS:
        r = c.post(LIST_URL, data={
            "draw": 1, "start": 0, "length": 500, "searchKeyword": kw,
            "searchCondition": "all", "searchReplyRegDateStart": "", "searchReplyRegDateEnd": "",
            "searchType": "", "searchCategory": "", "searchLawType": "",
        })
        rows = r.json().get("data", [])
        for row in rows:
            seen.setdefault(int(row["idx"]), row)
        print(f"  '{kw}' {len(rows)}건 (누적 {len(seen)})")
        time.sleep(DELAY)
    return seen


def fetch_detail(c: httpx.Client, idx: int, gubun: str) -> tuple[str, str] | None:
    if gubun == "법령해석":
        url = f"{LAW_DETAIL}?lawreqIdx={idx}&actCd=R&stNo=11&muNo=171"
    else:
        url = f"{OPINION_DETAIL}?opinionIdx={idx}&actCd=R&stNo=11&muNo=171"
    try:
        return clean(c.get(url).text), url
    except Exception as e:
        print(f"    idx={idx} 실패: {e}", file=sys.stderr)
        return None


ENDS = ("사이트 이용안내", "목록 URL 복사", "Family Site")


def parse(text: str, row: dict, url: str) -> Reply | None:
    dept_m = DEPT.search(text)
    dept = dept_m.group(1).strip() if dept_m else ""
    if KEEP_DEPT not in dept:
        return None
    q = section(text, "질의요지", ("회답",) + ENDS)
    a = section(text, "회답", ("이유",) + ENDS)
    r = section(text, "이유", ENDS)
    body = "\n\n".join(p for p in (
        f"[질의요지]\n{q}" if q else "",
        f"[회답]\n{a}" if a else "",
        f"[이유]\n{r}" if r else "",
    ) if p)
    if len(body) < 120:          # 본문이 사실상 비면 근거로 쓸 수 없다
        return None
    kind = row["gubun"]
    return Reply(
        id=f"금융위{kind}-{row['number']}",
        source=f"금융위원회 「{kind} 회신사례」 {row['regDate']} (일련번호 {row['number']})",
        source_url=url,
        doc_type=kind,
        topic=f"{dept} 소관",
        title=re.sub(r"\s+", " ", row["title"]).strip(),
        text=body,
        cited_articles=cited(body),
        reply_date=row["regDate"],
        department=dept,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="코퍼스에 병합")
    ap.add_argument("--limit", type=int, default=0, help="상세 조회 상한(시험용)")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    c = make_client()
    print("후보 수집 (검색어별):")
    cands = collect_candidates(c)
    items = sorted(cands.items())
    if args.limit:
        items = items[: args.limit]
    print(f"\n상세 조회 {len(items)}건 · 소관부서가 '{KEEP_DEPT}'인 것만 남긴다")

    kept: list[Reply] = []
    for n, (idx, row) in enumerate(items, 1):
        got = fetch_detail(c, idx, row["gubun"])
        time.sleep(DELAY)
        if not got:
            continue
        rep = parse(got[0], row, got[1])
        if rep:
            kept.append(rep)
            (RAW / f"{rep.id}.json").write_text(
                json.dumps(asdict(rep), ensure_ascii=False, indent=2), encoding="utf-8")
        if n % 50 == 0:
            print(f"  {n}/{len(items)} · 채택 {len(kept)}")

    print(f"\n채택 {len(kept)}건 / 조회 {len(items)}건")
    linked = sum(1 for r in kept if r.cited_articles)
    print(f"  금소법 조문 인용이 잡힌 건: {linked} ({linked/max(len(kept),1):.1%})")

    if not args.merge:
        print(f"\n원문을 {RAW}에 저장했다. 코퍼스 반영은 --merge.")
        return 0

    existing = [json.loads(l) for l in CORPUS.open(encoding="utf-8")] if CORPUS.exists() else []
    have = {r["id"] for r in existing}
    fields = ("id", "source", "source_url", "doc_type", "topic", "title", "text", "cited_articles")
    added = [{k: asdict(r)[k] for k in fields} for r in kept if r.id not in have]
    with CORPUS.open("a", encoding="utf-8") as f:
        for r in added:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n코퍼스에 {len(added)}건 추가 (기존 {len(existing)} → {len(existing)+len(added)})")
    print("이어서: python scripts/build_index.py && python scripts/build_graph.py")
    print("그리고 반드시: python scripts/eval_goldenset.py  — 문서를 늘리면 방해 문서도 는다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
