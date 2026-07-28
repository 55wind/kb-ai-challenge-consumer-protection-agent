"""그래프 근거 확장이 **새로운 질문**에서 작동하는지 측정.

골든셋으로 재면 안 된다. 골든셋 10문항은 조문 검색이 이미 95% 맞히는
질문이라 확장이 건질 것이 없다 — 실제 서비스는 우리가 예상하지 못한 질문을 받는다.

그래서 골든셋에 없는 소비자 표현으로 잰다. 특히 우리 코퍼스(금소법 중심)가
얇은 영역(투자상품·보험·약관)을 일부러 섞었다. 이런 질문에서 조문 검색은
엉뚱한 조문을 주거나 근거를 못 대는데, 같은 상황의 결정례는 자본시장법·상법을
인용하며 그 법리를 문장으로 서술해 두었다.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 기본 콘솔(cp949)에서 —·… 때문에 죽지 않게 한다
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.config import settings
from app.graph import expand
from app.graph.runtime import curated_graph
from app.search.retriever import NoEvidenceFound, _get_records, retrieve

# 골든셋에 없는 질문 — 실제 소비자 어투, 코퍼스가 얇은 영역 포함
NEW_QUERIES = [
    "증권사 직원이 제 성향이랑 안 맞는 상품을 계속 권했어요",
    "펀드 손실이 났는데 판매사가 책임이 전혀 없다고 합니다",
    "가입할 때 들은 설명이랑 실제 상품 내용이 달라요",
    "보험 가입할 때 알릴 의무를 몰랐는데 보험금을 못 준다고 해요",
    "약관에 저한테 불리한 조항이 있는데 무효 아닌가요",
    "직원이 수익률을 확정적으로 말했는데 손실이 났습니다",
    "고령인 아버지가 구조를 이해 못 하는 상품에 가입하셨어요",
    "신탁 상품인데 만기에 돈을 못 준다고 합니다",
    "계약서를 나중에 받았는데 내용이 설명과 다릅니다",
    "투자 위험을 알려주지 않아 손해를 봤는데 배상받을 수 있나요",
]


def main() -> int:
    graph = curated_graph()
    if graph is None:
        print("그래프가 없습니다. scripts/build_graph.py를 먼저 실행하세요.")
        return 1
    records = _get_records("laws")

    grounded = corpus_hit = no_eviden = 0
    total_added = 0
    print(f"새 질문 {len(NEW_QUERIES)}개 — 골든셋에 없는 표현\n")

    for q in NEW_QUERIES:
        try:
            hits = retrieve("laws", q, k=4)
            base = [h.record_id for h in hits]
        except NoEvidenceFound:
            hits, base = [], []
            no_eviden += 1

        try:
            cases = retrieve("cases", q, k=3)
        except NoEvidenceFound:
            cases = []
        titles = {h.record_id: getattr(h.record, "title", "") for h in cases}
        found = expand.cited_laws(graph, [h.record_id for h in cases],
                                  limit=settings.GRAPH_EXPANSION_MAX,
                                  exclude=set(base))

        added = []
        for c in found:
            if c.law_id in records:
                added.append((c.law_id, "코퍼스 조문", ""))
                corpus_hit += 1
            else:
                rec = expand.as_record(c, titles.get(c.cited_by, ""))
                if rec is not None:
                    added.append((c.law_id, "결정례 인용문", rec.text[:60]))
                    grounded += 1
        total_added += len(added)

        print(f"[{q[:38]}]")
        print(f"  검색 조문: {[b.split('-')[-1] for b in base[:3]] or '근거 없음'}")
        for lid, kind, snippet in added:
            print(f"  + {lid.replace('ref:', '')}  ({kind})")
            if snippet:
                print(f"      \"{snippet}…\"")
        if not added:
            print("  + (확장 없음)")
        print()

    print("─" * 60)
    print(f"질문 {len(NEW_QUERIES)}개 · 추가된 근거 {total_added}건")
    print(f"  결정례 인용문으로 근거화: {grounded}건")
    print(f"  코퍼스 조문 보강:        {corpus_hit}건")
    print(f"  조문 검색이 근거를 못 댄 질문: {no_eviden}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
