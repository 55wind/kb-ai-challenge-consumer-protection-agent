"""골든셋 검색 평가: hit@3 / hit@5.

골든셋은 실제 결정례·법령·절차·유권해석 문서 ID 기반이다 (창작 사례 없음).
질문은 소비자가 쓸 법한 말로 새로 쓰되, **정답 ID는 반드시 코퍼스에 실재하는
레코드**여야 한다. 이 스크립트는 그것을 먼저 검사한다.

왜 검사가 필요한가. 코퍼스가 갱신되면 결정례 번호가 바뀐다(`fss-제2021-9호-…`가
`fss-제2023-1호-…`로 재배정된 적이 있다). 정답 ID가 사라져도 평가는 조용히
계속 돌고, 그 문항은 **영원히 맞출 수 없는 문항**이 된다. 점수가 떨어지면
검색을 의심하게 되는데 원인은 골든셋에 있다. 그래서 먼저 실패시킨다.

기준: hit@5 >= 0.97. 이 서비스에서 검색 실패는 "근거 없음"으로 이어져 답변이
아예 나가지 않는다. 다섯 개를 보여주고도 못 찾는 일이 스무 번에 한 번이면
취약 소비자에게는 너무 잦다.
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
    hit3 = hit5 = 0
    misses = []
    for e in entries:
        try:
            ids = [h.record_id for h in retrieve(e["kind"], e["query"], k=5)]
        except NoEvidenceFound:
            ids = []
        expected = set(e["expected_ids"])
        got3 = bool(expected & set(ids[:3]))
        got5 = bool(expected & set(ids[:5]))
        hit3 += got3
        hit5 += got5
        per_kind[e["kind"]].append(got5)
        if not got5:
            misses.append((e["kind"], e["query"], ids[:3]))

    n = len(entries)
    print(f"골든셋 {n}문항 — hit@3: {hit3}/{n} ({hit3 / n:.1%}), "
          f"hit@5: {hit5}/{n} ({hit5 / n:.1%})  [기준 {TARGET_HIT5:.0%}]")
    for kind, results in sorted(per_kind.items()):
        ok = sum(results)
        print(f"  {kind:<16} hit@5 {ok}/{len(results)} ({ok / len(results):.0%})")
    if misses:
        print("\n놓친 문항:")
        for kind, q, got in misses:
            print(f"  [{kind}] {q[:44]}")
            print(f"        → {got}")
    return 0 if hit5 / n >= TARGET_HIT5 else 1


if __name__ == "__main__":
    sys.exit(main())
