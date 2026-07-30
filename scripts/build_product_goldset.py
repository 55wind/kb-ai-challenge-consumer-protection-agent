"""상품유형(금소법 제3조) 판정의 **구어체 정답지**를 만든다.

## 왜 필요한가

청약철회 기간은 제46조①이 상품유형별로 정한다(보장성 15일·투자성 7일·대출성 14일).
그래서 `product_type.infer`가 이 시스템에서 실제로 일하는 판정인데, 지금까지 그것을
잰 정답지는 **금감원 사례집의 사례 제목**이었다. 법률 문어체이고 공백조차 없다
(`만성위염치료사실미고지로인한계약해지의적정성여부`). 실사용 구어체와 분포가 다르다.

## 라벨을 우리가 붙이지 않는다

금소법 **제2조제1호 각 목**과 **제3조 각 호**가 상품 이름을 직접 열거한다.

    제2조1호 가·라목  예금, 대출          제3조1호 예금성 / 2호 대출성
    제2조1호 나목      금융투자상품        제3조3호 투자성
    제2조1호 다목      보험상품            제3조4호 보장성
    제2조1호 마목      신용카드, 시설대여, 연불판매, 할부금융   제3조2호 대출성
    시행령 제2조①     예탁금, 대부, 연계투자/연계대출, 신탁계약, 투자일임계약, 공제

**질의문에 이 낱말이 문자로 들어 있으면 상품유형은 조문이 정한다.** 우리가 문장을
읽고 판단하는 것이 아니라 조문을 적용하는 것이다. 두 유형 이상이 걸리는 문장은
제3조 단서(중복 귀속)에 해당하므로 정답을 하나로 정할 수 없어 **버린다**.

## 이 정답지의 한계 (반드시 함께 읽는다)

상품 이름이 **문자로 나온** 문항만 담긴다. 따라서 이것이 재는 것은
*"사용자가 상품을 말했을 때 맞히는가"*이고, *"말하지 않았을 때 추론하는가"*는
재지 않는다. 후자가 더 어렵고, 그 정확도는 **모른다**.

만들기: python scripts/build_product_goldset.py
재기:   python scripts/eval_product_type.py
"""
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402

OUT = settings.DATA_DIR / "product_goldset.jsonl"

# 조문이 열거한 상품 이름 → 제3조의 유형. 우리가 고른 낱말이 아니다.
STATUTE_NOUNS: dict[str, tuple[str, str]] = {
    "예금": ("예금성 상품", "법 제2조1호 가·라목 → 제3조1호"),
    "적금": ("예금성 상품", "영 제3조①2호(예금)에 준하는 수신 — 법 제3조1호"),
    "예탁금": ("예금성 상품", "영 제2조①2호 → 제3조1호"),
    "대출": ("대출성 상품", "법 제2조1호 가·라목 → 제3조2호"),
    "신용카드": ("대출성 상품", "법 제2조1호 마목 → 제3조2호"),
    "시설대여": ("대출성 상품", "법 제2조1호 마목 → 제3조2호"),
    "리스": ("대출성 상품", "법 제2조1호 마목 시설대여 → 제3조2호"),
    "연불판매": ("대출성 상품", "법 제2조1호 마목 → 제3조2호"),
    "할부금융": ("대출성 상품", "법 제2조1호 마목 → 제3조2호"),
    "대부": ("대출성 상품", "영 제2조①1호 → 제3조2호"),
    "금융투자상품": ("투자성 상품", "법 제2조1호 나목 → 제3조3호"),
    "신탁": ("투자성 상품", "영 제2조①4호 신탁계약 → 제3조3호"),
    "투자일임": ("투자성 상품", "영 제2조①4호 → 제3조3호"),
    "연계투자": ("투자성 상품", "영 제2조①3호 → 제3조3호"),
    "보험": ("보장성 상품", "법 제2조1호 다목 → 제3조4호"),
    "공제": ("보장성 상품", "영 제2조①2호 → 제3조4호"),
}

# '적금'은 '예금'을 부분문자열로 갖지 않지만, '보험료'·'대출금' 같은 파생어가
# 같은 유형을 가리키므로 부분 일치를 그대로 쓴다. 다만 다른 유형을 오염시키는
# 낱말은 제외한다 — '보험료 대출'(보험계약대출)은 두 유형이 겹치므로 버려진다.


def label(text: str) -> tuple[str, list[str]] | None:
    """조문 열거 낱말로 상품유형을 정한다. 둘 이상이면 None(제3조 단서)."""
    found: dict[str, list[str]] = {}
    for noun, (kind, basis) in STATUTE_NOUNS.items():
        if noun in text:
            found.setdefault(kind, []).append(f"'{noun}' — {basis}")
    if len(found) != 1:
        return None
    kind = next(iter(found))
    return kind, found[kind]


def candidates() -> list[tuple[str, str]]:
    """(질의, 출처). 소비자가 쓸 만한 **문장**만 모은다."""
    out: list[tuple[str, str]] = []
    # 1) 골든셋 — 검색 평가를 위해 소비자 말투로 쓴 질의
    gs = settings.DATA_DIR / "goldenset.jsonl"
    for r in (json.loads(x) for x in gs.open(encoding="utf-8")):
        out.append((r["query"], "골든셋(구어체 질의)"))
    # 2) 유권해석 코퍼스의 제목 중 **물음표로 끝나는 것** — FAQ·안내자료의 질문문
    interp = settings.CORPUS_DIR / "interpretations.jsonl"
    for r in (json.loads(x) for x in interp.open(encoding="utf-8")):
        t = r["title"].strip()
        if t.endswith("?") and 10 <= len(t) <= 120:
            out.append((t, f"{r['doc_type']} 제목({r['source'][:28]}…)"))
    return out


def main() -> int:
    seen: set[str] = set()
    rows = []
    dropped_multi = dropped_none = 0
    for q, src in candidates():
        q = re.sub(r"\s+", " ", q).strip()
        if q in seen:
            continue
        seen.add(q)
        got = label(q)
        if got is None:
            # 낱말이 하나도 없거나 둘 이상 — 어느 쪽이든 조문으로 정할 수 없다
            dropped_multi += 1
            continue
        kind, basis = got
        rows.append({"query": q, "product_type": kind, "label_basis": basis, "source": src})

    for q, _ in candidates():
        if label(re.sub(r"\s+", " ", q).strip()) is None:
            dropped_none += 1

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    import collections
    dist = collections.Counter(r["product_type"] for r in rows)
    print(f"{OUT.name}: {len(rows)}문항")
    for k, v in dist.most_common():
        print(f"  {v:4d}  {k}")
    print(f"  (조문 낱말이 없거나 둘 이상이라 버린 문항: {dropped_multi})")
    print()
    print("출처 분포:")
    for k, v in collections.Counter(r["source"].split("(")[0] for r in rows).most_common():
        print(f"  {v:4d}  {k}")
    print()
    print("표본:")
    for r in rows[:6]:
        print(f"  [{r['product_type']}] {r['query'][:64]}")
        print(f"     근거: {r['label_basis'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
