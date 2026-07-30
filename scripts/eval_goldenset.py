"""골든셋 검색 평가: hit@1 ~ hit@5.

골든셋은 실제 결정례·법령·절차·유권해석 문서 ID 기반이다 (창작 사례 없음).
질문은 소비자가 쓸 법한 말로 새로 쓰되, **정답 ID는 반드시 코퍼스에 실재하는
레코드**여야 한다. 이 스크립트는 그것을 먼저 검사한다.

왜 검사가 필요한가. 코퍼스가 갱신되면 결정례 번호가 바뀐다(`fss-제2021-9호-…`가
`fss-제2023-1호-…`로 재배정된 적이 있다). 정답 ID가 사라져도 평가는 조용히
계속 돌고, 그 문항은 **영원히 맞출 수 없는 문항**이 된다. 점수가 떨어지면
검색을 의심하게 되는데 원인은 골든셋에 있다. 그래서 먼저 실패시킨다.

**기준은 hit@4다.** 앱이 답변 경로에서 실제로 쓰는 k가 4이기 때문이다
(`orchestrator.py`의 `retrieve(kind, text, k=4)`). 예전에는 hit@3과 hit@5만 적었는데
어느 쪽도 앱이 쓰는 값이 아니었다 — hit@3이 낮게 나오면 검색을 의심하게 되지만
그 k로는 답변을 만들지 않고, hit@5는 실제보다 넉넉하다. 참고로 전 구간을 다 적는다.

이 서비스에서 검색 실패는 "근거 없음"으로 이어져 답변이 아예 나가지 않는다.
근거 넷을 보여주고도 못 찾는 일이 스무 번에 한 번이면 취약 소비자에게는 너무 잦다.
그래서 기준을 0.97로 둔다.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 한글·기호 출력이 Windows 기본 코드페이지(cp949)에서 죽지 않도록 강제.
# (심사·검증 환경이 대부분 Windows라 첫 실행이 UnicodeEncodeError로 끝나는 것을 막는다)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402
from app.corpus.loader import load_corpus  # noqa: E402
from app.search.retriever import NoEvidenceFound, retrieve  # noqa: E402

TARGET_HIT5 = 0.97


def _check_ids(entries: list[dict]) -> list[str]:
    """정답 ID가 코퍼스에 실재하는지 검사. 없는 ID를 문자열로 돌려준다."""
    known: dict[str, set[str]] = {}
    problems = []
    for e in entries:
        kind = e["kind"]
        if kind not in known:
            known[kind] = {r.id for r in load_corpus(kind)}
        missing = [i for i in e["expected_ids"] if i not in known[kind]]
        if missing:
            problems.append(f"  [{kind}] {e['query'][:38]}… → 없는 ID {missing}")
    return problems


def main() -> int:
    entries = [
        json.loads(l)
        for l in (settings.DATA_DIR / "goldenset.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    if problems := _check_ids(entries):
        print(f"골든셋에 실재하지 않는 정답 ID가 {len(problems)}건 있습니다 — "
              f"영원히 맞출 수 없는 문항입니다.")
        print("\n".join(problems))
        return 2

    per_kind: dict[str, list[int]] = defaultdict(list)
    # 앱이 답변 경로에서 실제로 쓰는 k는 4다(`orchestrator.py` — `retrieve(kind, text, k=4)`).
    # 예전에는 hit@3과 hit@5만 적었는데, 어느 쪽도 앱이 쓰는 값이 아니었다. hit@3이
    # 낮게 나오면 검색을 의심하게 되지만 그 k로는 답변을 만들지 않는다.
    APP_K = 4
    hits = {k: 0 for k in (1, 2, 3, 4, 5)}
    misses = []
    for e in entries:
        try:
            ids = [h.record_id for h in retrieve(e["kind"], e["query"], k=5)]
        except NoEvidenceFound:
            ids = []
        expected = set(e["expected_ids"])
        for k in hits:
            hits[k] += bool(expected & set(ids[:k]))
        got_app = bool(expected & set(ids[:APP_K]))
        per_kind[e["kind"]].append(got_app)
        if not got_app:
            rank = [i + 1 for i, x in enumerate(ids) if x in expected]
            misses.append((e["kind"], e["query"], ids[:APP_K], rank))

    n = len(entries)

    def ci(h: int) -> str:
        """윌슨 95% 신뢰구간. 100%에서도 구간이 1이 되지 않는다."""
        z, p = 1.96, h / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        s = (z * z / (4 * n * n) + p * (1 - p) / n) ** 0.5 * z / z
        s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
        return f"[{(c - s) / d:.1%}, {(c + s) / d:.1%}]"

    print(f"골든셋 {n}문항 — 기준 hit@{APP_K} >= {TARGET_HIT5:.0%}")
    for k in sorted(hits):
        mark = "  ← 앱이 답변에 쓰는 k" if k == APP_K else ""
        print(f"  hit@{k}  {hits[k]:2d}/{n} = {hits[k] / n:6.1%}   95%CI {ci(hits[k])}{mark}")
    print()
    for kind, results in sorted(per_kind.items()):
        ok = sum(results)
        print(f"  {kind:<16} hit@{APP_K} {ok}/{len(results)} ({ok / len(results):.0%})")
    if misses:
        print(f"\nhit@{APP_K}에서 놓친 문항:")
        for kind, q, got, rank in misses:
            print(f"  [{kind}] {q[:44]}")
            print(f"        정답 순위 {rank or '5위 밖'} · 상위: {got}")
    return 0 if hits[APP_K] / n >= TARGET_HIT5 else 1


if __name__ == "__main__":
    sys.exit(main())
