"""그래프 기반 근거 확장 — "검색을 연결로".

문제: 소비자는 법조문의 말로 묻지 않는다. "위험 설명을 못 들었어요"는
금소법 제19조(설명의무) 본문과 어휘도 의미도 잘 안 맞아 조문 검색에서 밀린다.
그런데 같은 상황을 다룬 **분쟁조정 결정례는 그 조문을 원문에서 인용**하고 있다.

그래서: 사용자 상황으로 결정례를 찾고 → 그 결정례가 실제로 인용한 조문을
근거에 더한다. 연결의 출처는 임베딩 추정이 아니라 **결정례 원문의 정규식 추출**
(cites 엣지, provenance=regex_citation)이므로 근거로 내세울 수 있다.

원칙:
- cites 엣지만 따른다. similar_to(임베딩 추정)는 근거 확장에 쓰지 않는다.
- 담당자가 끈 엣지는 따르지 않는다 — 큐레이션이 실제로 답변에 반영되는 지점.
- 그래프가 없으면 아무 일도 하지 않는다(비파괴).
"""
import re
from dataclasses import dataclass

from app.graph.context_graph import ContextGraph


@dataclass(frozen=True)
class CitedLaw:
    """결정례가 원문에서 인용한 조문."""

    law_id: str
    cited_by: str      # 어느 결정례가 인용했는가 (설명·감사용)
    evidence: str      # 인용이 발견된 원문 스니펫
    label: str = ""    # 조문 표시명 (stub 노드의 label)
    source: str = ""   # 법령 출처 (stub 노드 meta)
    source_url: str = ""
    is_stub: bool = False  # 코퍼스에 본문이 없는 참조 법령


@dataclass(frozen=True)
class CitedLawRecord:
    """코퍼스 밖 법령을 근거로 쓰기 위한 레코드.

    조문 본문은 우리에게 없다. 대신 **분쟁조정위원회가 그 조문의 법리를
    서술한 문장**이 결정례 원문에 있다. 그 문장을 근거로 삼고 출처를
    결정례로 명시한다 — 갖지 않은 법령 본문을 가진 척하지 않는다.

    answerer는 레코드를 덕타이핑으로 읽으므로(article/text/source/source_url)
    코퍼스 레코드와 같은 자리에 들어갈 수 있다.
    """

    id: str
    article: str
    text: str
    source: str
    source_url: str


def cited_laws(
    graph: ContextGraph | None,
    case_ids: list[str],
    limit: int = 2,
    exclude: set[str] | None = None,
) -> list[CitedLaw]:
    """결정례들이 인용한 조문을, 여러 결정례가 함께 인용한 순으로 반환.

    여러 사례가 같은 조문을 인용했다면 그 유형의 핵심 쟁점일 가능성이 높다.
    그래서 인용 횟수를 우선순위로 쓴다.
    """
    if graph is None or not case_ids:
        return []
    skip = exclude or set()

    votes: dict[str, int] = {}
    first: dict[str, CitedLaw] = {}
    for cid in case_ids:
        if cid not in graph.nodes:
            continue
        for edge, other in graph.neighbors(cid, rel="cites"):
            # neighbors는 양방향을 준다 — 결정례가 '인용한' 방향만 취한다
            if edge.src != cid:
                continue
            if other.id in skip:
                continue
            votes[other.id] = votes.get(other.id, 0) + 1
            first.setdefault(
                other.id,
                CitedLaw(
                    law_id=other.id,
                    cited_by=cid,
                    evidence=str(edge.provenance.get("evidence", "")),
                    label=other.label,
                    source=str(other.meta.get("source", "")),
                    source_url=str(other.meta.get("source_url", "")),
                    is_stub=bool(other.meta.get("stub")),
                ),
            )
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    return [first[lid] for lid, _ in ranked[:limit]]


# 인용 문장이 이보다 짧으면 법리가 담기지 않는다 — 근거로 쓰지 않는다
_MIN_EVIDENCE = 40
_ARTICLE = re.compile(r"제\s*\d+\s*조")


def _clean_snippet(raw: str) -> str:
    """고정 폭으로 잘린 인용 창을 문장 경계로 다듬는다.

    추출은 인용 위치 앞뒤 40자를 그대로 떠오므로 "익증권 매매계약 취소 관련"처럼
    단어 중간에서 시작하는 조각이 나온다. 조각을 근거랍시고 LLM에 넣으면
    엉뚱한 답이 나오므로, 첫 문장 경계 이후부터 쓴다.
    """
    s = " ".join(raw.split())
    # 앞쪽 조각 제거: 첫 문장 끝(. 다./며/고 등) 뒤부터
    m = re.search(r"[.。]\s+|(?<=다)\s{2,}", s)
    if m and m.end() < len(s) * 0.5:
        s = s[m.end():]
    return s.strip(" ·,)】]")


def as_record(c: CitedLaw, case_title: str = "") -> CitedLawRecord | None:
    """코퍼스 밖 인용을 근거 레코드로. 근거가 될 수 없으면 None.

    두 가지를 요구한다.
    1. 충분한 길이 — 조각에는 법리가 담기지 않는다
    2. 문장 안에 조문 번호가 실제로 있을 것 — 인용 위치 근처를 떠온 것이지
       그 조문을 설명하는 문장이 아닐 수 있다. 번호가 없으면 무엇에 대한
       설명인지 확인할 수 없으므로 근거로 쓰지 않는다.
    """
    text = _clean_snippet(c.evidence)
    if len(text) < _MIN_EVIDENCE or not _ARTICLE.search(text):
        return None
    cited_by = case_title or c.cited_by
    return CitedLawRecord(
        id=c.law_id,
        article=c.label or c.law_id,
        text=text,
        # 출처를 결정례로 명시한다 — 법령 본문을 인용한 것이 아니다
        source=f"{c.source} — 금감원 결정례 「{cited_by[:34]}」가 인용",
        source_url=c.source_url,
    )
