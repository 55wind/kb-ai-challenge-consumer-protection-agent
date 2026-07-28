"""골든셋 검색 평가: hit@3 / hit@5.

골든셋은 실제 결정례·법령·절차 문서 ID 기반 (창작 사례 없음).
기준: hit@5 >= 0.8 미달 시 임계값·청킹 튜닝 필요.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 한글·기호 출력이 Windows 기본 코드페이지(cp949)에서 죽지 않도록 강제.
# (심사·검증 환경이 대부분 Windows라 첫 실행이 UnicodeEncodeError로 끝나는 것을 막는다)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings
from app.search.retriever import NoEvidenceFound, retrieve


def main() -> int:
    entries = [
        json.loads(l)
        for l in (settings.DATA_DIR / "goldenset.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    hit3 = hit5 = 0
    misses = []
    for e in entries:
        try:
            hits = retrieve(e["kind"], e["query"], k=5)
            ids = [h.record_id for h in hits]
        except NoEvidenceFound:
            ids = []
        expected = set(e["expected_ids"])
        if expected & set(ids[:3]):
            hit3 += 1
        if expected & set(ids[:5]):
            hit5 += 1
        else:
            misses.append((e["query"], ids[:3]))
    n = len(entries)
    print(f"골든셋 {n}문항 — hit@3: {hit3}/{n} ({hit3/n:.0%}), hit@5: {hit5}/{n} ({hit5/n:.0%})")
    for q, got in misses:
        print(f"  [miss] {q[:40]} → {got}")
    return 0 if hit5 / n >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
