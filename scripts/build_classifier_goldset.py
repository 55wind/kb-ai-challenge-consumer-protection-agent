"""분류 평가의 정답지를 **금융감독원 분류**로 다시 만든다.

**무엇이 문제였나.**

`cases.jsonl`의 `category`는 금감원이 붙인 값이 아니다. `build_corpus.py`의
`_category(title, body)` — 우리가 짠 키워드 규칙이 만든 값이다. 그런데 문서들은
그것을 "기관이 분류한 실데이터"라고 적어 왔다. 사실이 아니었고, 실제로 틀려
있었다. 금감원 사례집과 제목이 이어지는 17건을 대조하니 이런 것들이 나왔다.

    「계약전 발병 부담보 조항의 유효성 여부」
        우리 = 대출·금리        금감원 = 계약전 알릴의무 위반 (보험)
    「포괄적 일임을 받은 증권사 직원이 퇴사한 이후 …」
        우리 = 보이스피싱·전자금융사기   금감원 = 임의매매

**무엇으로 바꾸나.**

금융감독원 「금융분쟁조정 사례집」 두 권의 사례 목차에는 조정사례 410건에
금감원이 직접 붙인 3단계 분류(편-장-절)가 있다(`build_fss_case_index.py`).
그 장(章)을 우리 유형 체계로 옮겨 정답지를 만든다.

**대응표는 판단이다 — 다만 한 번만 하는 판단이다.**

금감원 분류와 우리 유형 체계는 목적이 다르다. 금감원 것은 분쟁 주제별 편람
분류이고, 우리 것은 **법정 기한 룰과 절차 경로를 고르기 위한** 분류다. 그래서
1:1로 겹치지 않는다. 아래 대응표는 그 차이를 메우는 판단이고, 사례마다 다시
판단하는 키워드 규칙과 달리 **19개 장에 대해 한 번 정하고 기록**한다.

겹치지 않는 자리도 그대로 적는다.

  - 금감원 `설명의무위반및부당권유`는 우리 체계에서 **불완전판매**와
    **부당권유·불공정영업** 둘로 갈린다. 장 단위로는 가를 수 없어 불완전판매로
    보낸다 — 사례집 절(節)이 전부 상품 판매 국면이기 때문이다. 이 때문에
    부당권유·불공정영업은 이 정답지로 검증되지 않는다.
  - `보이스피싱·전자금융사기`, `절차문의`에 대응하는 장이 사례집에 없다.
    이 두 유형은 여전히 검증 수단이 없다.

**정답지의 한계.**

입력이 **사례 제목**이다. 실제 소비자가 쓰는 구어체 민원과 문체가 다르다.
결정례 `facts`를 쓰던 이전 정답지와 같은 한계이며, 여기서도 대리 지표다.
다만 **라벨의 출처가 우리에서 금감원으로 바뀐 것**이 이 작업의 요점이다.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.agent.classifier import CATEGORIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "fss_case_index.jsonl"
OUT = ROOT / "data" / "classifier_goldset.jsonl"

# 금감원 사례집 장(章) → 이 서비스의 민원 유형.
# 값이 None이면 우리 체계로 옮길 수 없다는 뜻이고, 정답지에서 제외한다.
CHAPTER_MAP: dict[str, str | None] = {
    # ── 보험편 ─────────────────────────────────────────────
    "계약전알릴의무위반및계약해지등": "보험",
    "재해등인정여부": "보험",
    "암진단등": "보험",
    "장해및수술해당여부": "보험",
    "법률및약관해석등": "보험",
    "자동차보험": "보험",
    "질병․상해등보험": "보험",
    "보증및배상책임보험": "보험",
    "기타": "보험",  # 보험편의 '기타' — 운전자·화재·해상·건설공사 보험
    # ── 은행·중소서민금융·금융투자편 ─────────────────────────
    "대출거래": "대출·금리",
    "담보책임": "대출·금리",
    "보증책임": "대출·금리",
    "예금등수신거래": "예금·계좌",
    "신용카드등": "카드",
    "금융투자상품판매": "불완전판매",
    "설명의무위반및부당권유": "불완전판매",
    "매매주문관련": "투자상품",
    "임의매매": "투자상품",
    # 전산장애·사무처리 오류는 우리 10개 유형 어디에도 맞지 않는다.
    # '기타'로 몰면 평가가 무의미해지므로 정답지에서 뺀다.
    "전산장애,사무처리오류등": None,
}


def main() -> int:
    if not INDEX.exists():
        print(f"먼저 build_fss_case_index.py를 실행하세요: {INDEX} 없음")
        return 2

    entries = [json.loads(x) for x in INDEX.read_text(encoding="utf-8").splitlines() if x.strip()]

    unmapped = {e["chapter"] for e in entries if e["chapter"] and e["chapter"] not in CHAPTER_MAP}
    if unmapped:
        print(f"대응표에 없는 장이 있습니다 — 확인이 필요합니다: {sorted(unmapped)}")
        return 2

    rows, dropped = [], collections.Counter()
    for e in entries:
        cat = CHAPTER_MAP.get(e["chapter"])
        if cat is None:
            dropped[e["chapter"] or "(장 없음)"] += 1
            continue
        assert cat in CATEGORIES, f"대응표가 존재하지 않는 유형을 가리킵니다: {cat}"
        rows.append({
            "query": e["title"],
            "category": cat,
            "fss_chapter": e["chapter"],
            "fss_section": e["section"],
            "year": e["year"],
            "source": e["source"],
            "source_url": e["source_url"],
        })

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")

    dist = collections.Counter(r["category"] for r in rows)
    print(f"금감원 사례집 {len(entries)}건 → 정답지 {len(rows)}건  → {OUT}")
    if dropped:
        print(f"제외: {dict(dropped)} (우리 유형 체계로 옮길 수 없는 장)")
    print("\n유형 분포")
    for c in CATEGORIES:
        n = dist.get(c, 0)
        mark = "" if n else "   ← 이 정답지로는 검증되지 않는다"
        print(f"  {c:<20} {n:>4}건{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
