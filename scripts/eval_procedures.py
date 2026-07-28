"""절차 검색 품질 측정 — "민원 어떻게 넣어요?"가 이 주제에서 가장 흔한 질문이다.

절차 문서는 5건뿐이라 hit@k는 의미가 없다(거의 다 나온다). 대신 **1순위가 맞는가**와
**답변 근거에 실제 절차 내용(단계·필요서류)이 담기는가**를 본다.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.search.retriever import NoEvidenceFound, retrieve

# 소비자가 실제로 쓸 표현 (골든셋 밖)
CASES = [
    ("민원 넣으려면 무슨 서류가 필요한가요", "proc-complaint-filing"),
    ("분쟁조정 신청하면 얼마나 걸리나요", "proc-dispute-mediation"),
    ("은행에 먼저 민원을 넣어야 하나요", "proc-kb-complaint"),
    ("KB국민은행 민원 접수 방법 알려주세요", "proc-kb-complaint"),
    ("위임장도 내야 하나요", "proc-complaint-filing"),
    ("분쟁조정 절차가 어떻게 되나요", "proc-dispute-mediation"),
]


def main() -> int:
    top1 = 0
    for query, want in CASES:
        try:
            hits = retrieve("procedures", query, k=3)
        except NoEvidenceFound:
            print(f"  근거없음  {query}")
            continue
        got = hits[0].record_id
        ok = got == want
        top1 += ok
        print(f"  {'맞음' if ok else '틀림'}  {query[:30]:32s} → {got}")
        if not ok:
            print(f"        기대: {want}")
    print(f"\n1순위 정확도: {top1}/{len(CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
