"""컨텍스트 그래프 빌드: 코퍼스·룰 테이블 → data/indexes/graph.json.

모든 엣지는 결정론 추출 + provenance 부착 (LLM 불사용, semantica 원칙).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.agent.classifier import CATEGORIES
from app.agent.deadlines import RULES as DEADLINE_RULES
from app.agent.rules.sales_rules import RULES as REDFLAG_RULES
from app.corpus.loader import load_corpus
from app.graph.citations import (
    extract_law_citations,
    extract_stub_citations,
    stub_law_url,
)
from app.graph.context_graph import ContextGraph, Node

_NOW = datetime.now(timezone.utc).isoformat()

# 유형 → 구제 절차 정적 매핑 (실제 절차 문서 id)
_CATEGORY_PROCEDURES = {
    "보이스피싱·전자금융사기": ["proc-kb-complaint", "proc-dispute-mediation"],
    "불완전판매": ["proc-kb-complaint", "proc-dispute-mediation"],
    "부당권유·불공정영업": ["proc-kb-complaint", "proc-dispute-mediation"],
    "대출·금리": ["proc-kb-complaint", "proc-complaint-filing"],
    "카드": ["proc-kb-complaint", "proc-complaint-filing"],
    "보험": ["proc-dispute-mediation"],
    "예금·계좌": ["proc-kb-complaint"],
    "투자상품": ["proc-kb-complaint", "proc-dispute-mediation", "proc-other-bodies"],
    "절차문의": ["proc-complaint-filing"],
}

_SIM_THRESHOLD = 0.75
_SIM_TOP_K = 3


def _prov(method: str, source_id: str, evidence: str = "") -> dict:
    return {"method": method, "source_id": source_id, "evidence": evidence[:200], "created_at": _NOW}


def _retrieval_top1(kind: str, query: str):
    from app.search.retriever import NoEvidenceFound, retrieve

    try:
        return retrieve(kind, query, k=1)[0]
    except NoEvidenceFound:
        return None


def build() -> ContextGraph:
    g = ContextGraph()
    laws = load_corpus("laws")
    cases = load_corpus("cases")
    procedures = load_corpus("procedures")
    products = load_corpus("kb_products")

    # ── 노드 ──
    for name in CATEGORIES:
        g.add_node(Node(id=f"cat:{name}", type="category", label=name))
    for r in laws:
        g.add_node(Node(id=r.id, type="law", label=f"{r.law_name.split(' ')[0]} {r.article}",
                        meta={"source": r.source, "source_url": r.source_url}))
    for r in cases:
        g.add_node(Node(id=r.id, type="case", label=r.title[:40],
                        meta={"decision": r.decision, "source": r.source, "source_url": r.source_url}))
    for r in procedures:
        g.add_node(Node(id=r.id, type="procedure", label=r.title,
                        meta={"source": r.source, "source_url": r.source_url}))
    product_docs = sorted({(r.product_name, r.doc_type) for r in products})
    for pname, dtype in product_docs:
        g.add_node(Node(id=f"prod:{pname}", type="product", label=f"{pname} ({dtype})",
                        meta={"source": f"KB국민은행 {pname}"}))
    for rule in REDFLAG_RULES:
        g.add_node(Node(id=f"rule:{rule.name}", type="rule", label=rule.name))
    for cat, rules in DEADLINE_RULES.items():
        for rule in rules:
            g.add_node(Node(id=f"dl:{rule.action[:40]}", type="deadline", label=rule.action[:40],
                            meta={"period": rule.period}))

    # ── 엣지 1: case -cites-> law (원문 인용 정규식) ──
    # 코퍼스 보유 법령은 조문 노드로, 그 외(민법·자본시장법 등)는 law.go.kr
    # 딥링크가 달린 참조(stub) 노드로 연결 — 인용은 전부 결정례 원문의 사실.
    n_cites = 0
    for r in cases:
        body = f"{r.facts}\n{r.reasoning}"
        for chunk_id, evidence in extract_law_citations(body):
            if g.add_edge(r.id, "cites", chunk_id, _prov("regex_citation", r.id, evidence)):
                n_cites += 1
        for stub_id, law_name, article, evidence in extract_stub_citations(body):
            if stub_id not in g.nodes:
                g.add_node(Node(id=stub_id, type="law", label=f"{law_name.split(' ')[0]} {article}",
                                meta={"stub": True, "source": f"국가법령정보센터 {law_name}",
                                      "source_url": stub_law_url(law_name, article)}))
            if g.add_edge(r.id, "cites", stub_id, _prov("regex_citation", r.id, evidence)):
                n_cites += 1

    # ── 엣지 2: case -categorized_as-> category ──
    for r in cases:
        g.add_edge(r.id, "categorized_as", f"cat:{r.category}",
                   _prov("corpus_label", r.id)) if f"cat:{r.category}" in g.nodes else None

    # ── 엣지 3: case -similar_to-> case (FAISS 벡터 재구성 코사인) ──
    from app.search.indexer import load_index

    idx = load_index("cases")
    vecs = np.stack([idx.faiss_index.reconstruct(i) for i in range(len(idx.ids))])
    sims = vecs @ vecs.T
    for i, src_id in enumerate(idx.ids):
        order = np.argsort(sims[i])[::-1]
        added = 0
        for j in order:
            if j == i or added >= _SIM_TOP_K:
                continue
            if sims[i][j] < _SIM_THRESHOLD:
                break
            g.add_edge(src_id, "similar_to", idx.ids[j],
                       _prov("embedding_similarity", src_id, f"cosine={sims[i][j]:.3f}"))
            added += 1

    # ── 엣지 4: rule/deadline -grounded_in-> law (빌드 타임 retrieve 고정) ──
    for rule in REDFLAG_RULES:
        hit = _retrieval_top1("laws", rule.law_query)
        if hit:
            g.add_edge(f"rule:{rule.name}", "grounded_in", hit.record_id,
                       _prov("retrieval_top1", f"rule:{rule.name}", rule.law_query))
    for cat, rules in DEADLINE_RULES.items():
        for rule in rules:
            hit = _retrieval_top1("laws", rule.basis_query)
            if hit:
                g.add_edge(f"dl:{rule.action[:40]}", "grounded_in", hit.record_id,
                           _prov("retrieval_top1", f"dl:{rule.action[:40]}", rule.basis_query))

    # ── 엣지 5: deadline -applies_to-> category (룰 테이블) ──
    for cat, rules in DEADLINE_RULES.items():
        targets = list(CATEGORIES) if cat == "_common" else [cat]
        for rule in rules:
            for t in targets:
                if f"cat:{t}" in g.nodes:
                    g.add_edge(f"dl:{rule.action[:40]}", "applies_to", f"cat:{t}",
                               _prov("rule_table", "deadlines.RULES"))

    # ── 엣지 6: category -remedied_by-> procedure (정적 매핑) ──
    for cat, procs in _CATEGORY_PROCEDURES.items():
        for pid in procs:
            if f"cat:{cat}" in g.nodes and pid in g.nodes:
                g.add_edge(f"cat:{cat}", "remedied_by", pid, _prov("static_map", "build_graph"))

    # ── 엣지 7: procedure -escalates_to-> procedure (2단계 민원 경로) ──
    g.add_edge("proc-kb-complaint", "escalates_to", "proc-dispute-mediation",
               _prov("static_map", "2단계 민원 경로"))

    # ── 엣지 8: product -governed_by-> law (약관 원문 인용) ──
    for r in products:
        for chunk_id, evidence in extract_law_citations(r.text):
            g.add_edge(f"prod:{r.product_name}", "governed_by", chunk_id,
                       _prov("regex_citation", r.id, evidence))

    print(f"cites 엣지: {n_cites}")
    return g


if __name__ == "__main__":
    g = build()
    path = g.save()
    s = g.stats()
    print(f"그래프 저장 → {path}")
    print(f"노드 {s['nodes']} ({s['nodes_by_type']})")
    print(f"엣지 {s['edges']} ({s['edges_by_rel']})")
