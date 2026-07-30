"""금감원 사례집이 닿지 않는 유형의 프로브를 **실제 자료에서** 만든다.

**왜 다시 만드나.** 이전 프로브(`data/classifier_probe.jsonl`)는 골든셋 질의를
가져와 **우리가 유형을 붙인 것**이었다. 그것은 분류 정답지에서 방금 걷어낸
문제와 같은 문제다 — 우리가 만든 라벨로 우리 분류기를 재는 것. 그래서 버리고
라벨을 감독당국 쪽으로 옮긴다.

**라벨을 어디서 얻나 — "그 문서가 어느 장에 실려 있는가".**

금감원 사례집에서 쓴 것과 같은 원리다. 사람이 문장을 읽고 판단하는 것이 아니라,
**감독당국이 그 문서를 어느 분류 아래 넣었는지**를 라벨로 삼는다.

  `부당권유·불공정영업`
      금융위원회 「금소법 FAQ 답변(3차)」의 두 章 —
      「Ⅰ. 불공정영업행위 관련」, 「Ⅱ. 특정 영업행위의 '권유'여부 판단 관련」.
      금융위가 그 章에 넣은 문항이므로 라벨이 금융위 것이다.

  `절차문의`
      금감원 민원·분쟁조정 **절차 안내 페이지** 자체(`procedures` 코퍼스).
      금감원이 '민원 절차' 메뉴에 게시한 문서이므로 유형이 문서의 소재로 정해진다.

**못 만든 것도 적는다.**

  `보이스피싱·전자금융사기`
      사례집에 이 이름의 장이 없다. 가까운 절이 「신용카드의 도용 및 위·변조 사용」
      「신용카드의 제3자 부당사용」인데, **우리 분류 체계가 여기서 겹친다** —
      `카드` 설명에도 "카드 부정사용", `보이스피싱·전자금융사기` 설명에도
      "명의도용"이 있어 어느 쪽이 정답인지 우리 스스로 정하지 못한다.
      억지로 한쪽에 밀어 넣으면 또 우리 판단이 라벨이 된다. 그래서 비운다.
      **이것은 데이터가 없는 문제가 아니라 분류 체계의 결함이다.**

  `기타`
      정의상 대응하는 실제 자료가 없다.

따라서 실제 라벨로 검증되는 유형은 **10종 중 8종**이다(사례집 6 + 이 프로브 2).
나머지 2종의 정확도는 **모른다고 적는다.**
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.agent.classifier import CATEGORIES  # noqa: E402
from app.config import settings  # noqa: E402
from app.corpus.loader import load_corpus  # noqa: E402

OUT = settings.DATA_DIR / "classifier_probe.jsonl"

# 금융위 FAQ의 章 → 우리 유형. 章 제목이 우리 유형 이름과 거의 그대로 겹치는
# 것만 쓴다. 애매한 章(등록요건·내부통제기준·소비자 권리 등)은 넣지 않는다.
FAQ_TOPIC_MAP = {
    "Ⅰ. 불공정영업행위 관련": "부당권유·불공정영업",
    "Ⅱ. 특정 영업행위의 ‘권유’여부 판단 관련": "부당권유·불공정영업",
}


def main() -> int:
    rows = []

    for r in load_corpus("interpretations"):
        cat = FAQ_TOPIC_MAP.get(r.topic)
        if cat and r.doc_type == "FAQ":
            rows.append({
                "query": r.title,
                "category": cat,
                "label_basis": f"{r.source} — {r.topic}",
                "source_url": r.source_url,
            })

    for r in load_corpus("procedures"):
        rows.append({
            "query": r.title,
            "category": "절차문의",
            "label_basis": f"{r.source} (민원·분쟁조정 절차 안내 문서)",
            "source_url": r.source_url,
        })

    for r in rows:
        assert r["category"] in CATEGORIES, r["category"]

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")

    import collections
    dist = collections.Counter(r["category"] for r in rows)
    print(f"프로브 {len(rows)}건 → {OUT}")
    for c, n in dist.most_common():
        print(f"  {c:<20} {n}건")
    print("\n라벨 근거는 전부 감독당국 문서의 소재다. 우리가 붙인 라벨은 없다.")
    print("비운 유형: 보이스피싱·전자금융사기(분류 체계가 '카드'와 겹침), 기타(해당 자료 없음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
