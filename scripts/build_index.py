"""실코퍼스 3종 → 하이브리드 인덱스 생성."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.corpus.loader import load_corpus
from app.search.indexer import build_index

def _procedure_text(r) -> str:
    """절차 문서는 단계·필요서류까지 넣어야 검색된다.

    text는 167~397자로 개요뿐이고, 정작 소비자가 묻는 것("무슨 서류가 필요한가요",
    "얼마나 걸리나요")의 답은 steps·required_documents에 있다. 이 둘을 빼고
    인덱싱하면 "위임장도 내야 하나요" 같은 질문이 근거 없음으로 빠진다.
    """
    parts = [r.title, r.text]
    if r.steps:
        parts.append("처리 절차 단계: " + " ".join(r.steps))
    if r.required_documents:
        parts.append("필요 서류: " + " ".join(r.required_documents))
    return "\n".join(parts)


TEXT_FNS = {
    "laws": lambda r: f"{r.law_name} {r.article}\n{r.text}",
    "cases": lambda r: f"{r.title}\n{r.facts}\n{r.issues}",
    "procedures": _procedure_text,
    "kb_products": lambda r: f"{r.product_name} {r.doc_type} {r.section}\n{r.text}",
}


def main() -> None:
    for kind, fn in TEXT_FNS.items():
        records = load_corpus(kind)
        t0 = time.time()
        build_index(kind, records, fn)
        print(f"{kind}: {len(records)}건 인덱싱 완료 ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
