"""검색기 두 개(벡터·BM25)를 각각, 그리고 섞은 비율별로 재 본다.

**왜 이 측정이 필요한가.**

RRF를 같은 가중치로 합치는 것은 "두 검색기가 비슷하게 믿을 만하다"는 가정이다.
이 서비스에서는 그 가정이 성립하지 않을 수 있다. 사용자는 "상담해준 사람이 은행
직원인지 대리점인지 밝히지 않았어요"라고 쓰지, "고지의무"라고 쓰지 않는다.
BM25는 겹치는 낱말을 찾는데, 조문에는 그 낱말이 없다. 대신 "금융상품·판매·대리·
중개" 같은 흔한 낱말이 **등록요건 조문**(교육·경력 요건을 길게 나열한 조문)에
잔뜩 있어서 그쪽이 올라온다.

가정을 검증하지 않고 가중치를 손대면 골든셋에 맞춰 깎는 것(overfitting)이 된다.
그래서 먼저 검색기를 따로 재고, 차이가 실제로 있는지 본 다음에 비율을 정한다.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402
from app.search import embedder  # noqa: E402
from app.search.indexer import tokenize  # noqa: E402
from app.search.retriever import _get_index  # noqa: E402

# 조정 전 값. 표에서 '현재'와 비교 기준으로 쓴다.
_BASELINE_RRF_K = 60


def rrf(vec_hits, bm_hits, k, w_vec=1.0, w_bm=1.0, rrf_k=_BASELINE_RRF_K):
    scores = defaultdict(float)
    for rank, (rid, _) in enumerate(vec_hits):
        scores[rid] += w_vec / (rrf_k + rank + 1)
    for rank, (rid, _) in enumerate(bm_hits):
        scores[rid] += w_bm / (rrf_k + rank + 1)
    return [r for r, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


def main() -> int:
    entries = [
        json.loads(line)
        for line in (settings.DATA_DIR / "goldenset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # 질의별 후보를 한 번만 뽑아 두고 비율만 바꿔 가며 잰다 — 임베딩을 다시
    # 돌리면 한 조합에 수 분이 걸려 비교 자체가 불가능해진다.
    cached = []
    for e in entries:
        index = _get_index(e["kind"])
        qv = embedder.embed([e["query"]])[0]
        cached.append((
            e,
            index.search_vec(qv, k=15),
            index.search_bm25(tokenize(e["query"]), k=15),
        ))
    print(f"골든셋 {len(entries)}문항 · 후보 캐시 완료\n")

    def score(w_vec, w_bm, rrf_k=_BASELINE_RRF_K):
        per_kind = defaultdict(lambda: [0, 0])
        total = 0
        for e, v, b in cached:
            ids = rrf(v, b, 5, w_vec, w_bm, rrf_k)
            ok = bool(set(e["expected_ids"]) & set(ids))
            per_kind[e["kind"]][0] += ok
            per_kind[e["kind"]][1] += 1
            total += ok
        return total, per_kind

    n = len(cached)

    def row(label, wv, wb, rk=_BASELINE_RRF_K):
        total, pk = score(wv, wb, rk)
        cells = "".join(
            f"{pk[k][0]}/{pk[k][1]}".ljust(11)
            for k in ("laws", "cases", "interpretations", "procedures")
        )
        print(f"{label:<24}{f'{total}/{n} ({total / n:.1%})':<15}{cells}")

    header = f"{'조합':<24}{'전체':<15}{'laws':<11}{'cases':<11}{'interp':<11}{'proc':<11}"
    print("① 검색기 비중 (RRF K=60 고정)")
    print(header)
    for label, wv, wb in [("벡터만", 1.0, 0.0), ("BM25만", 0.0, 1.0),
                          ("1 : 1 (현재)", 1.0, 1.0), ("1 : 0.5", 1.0, 0.5)]:
        row(label, wv, wb)

    # RRF의 K는 원 논문에서 수백 건짜리 목록을 합칠 때의 값(60)이다. 여기서는
    # 후보가 15개뿐이라 1위(1/61)와 5위(1/65)의 차이가 6%밖에 안 난다. 순위가
    # 사실상 무의미해지고, 두 목록에 겹쳐 나오기만 하면 이긴다 — 정답이 벡터
    # 3위인데도 밀리는 이유가 이것이다. K를 후보 수에 맞춰 낮춰 본다.
    print("\n② RRF K (비중 1 : 0.5 고정)")
    print(header)
    for rk in (60, 30, 20, 10, 5, 2):
        row(f"K = {rk}", 1.0, 0.5, rk)

    print("\n③ 비중 × K 조합")
    print(header)
    for rk in (20, 10, 5):
        for wb in (1.0, 0.5, 0.25):
            row(f"K={rk}, 1 : {wb}", 1.0, wb, rk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
