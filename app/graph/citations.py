"""결정례·약관 원문에서 법령 인용을 결정론적으로 추출 (LLM 불사용).

코퍼스에 있는 법령만 해석한다 — 매핑 불가한 인용(예: 자본시장법)은 버린다.
"""
import re

# 코퍼스 보유 법령 (조문 전문 있음)
_ALIAS = [
    (r"금융소비자\s*보호에\s*관한\s*법률|금융소비자보호법|금소법", "금융소비자 보호에 관한 법률"),
    (r"전기통신금융사기\s*피해[^제]{0,30}특별법|통신사기피해환급법", "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법"),
]

# 코퍼스 밖 법령 — 인용은 실제이므로 law.go.kr 딥링크 참조(stub) 노드로 연결
STUB_LAWS = [
    (r"민법", "민법"),
    (r"자본시장과\s*금융투자업에\s*관한\s*법률|자본시장법", "자본시장과 금융투자업에 관한 법률"),
    (r"상법", "상법"),
    (r"약관의\s*규제에\s*관한\s*법률|약관규제법", "약관의 규제에 관한 법률"),
    (r"할부거래에\s*관한\s*법률|할부거래법", "할부거래에 관한 법률"),
    (r"보험업법", "보험업법"),
]
_SUFFIX = {"시행령": " 시행령", "감독규정": "에 관한 감독규정"}

_CITE = re.compile(
    r"(?P<alias>" + "|".join(a for a, _ in _ALIAS) + r")"
    r"\s*(?P<suffix>시행령|감독규정)?"
    r"[^가-힣0-9]{0,10}?제\s*(?P<no>\d+)\s*조(?:\s*의\s*(?P<sub>\d+))?"
)
_STUB_CITE = re.compile(
    r"(?P<alias>" + "|".join(a for a, _ in STUB_LAWS) + r")"
    r"\s*(?P<suffix>시행령)?"
    r"[^가-힣0-9]{0,10}?제\s*(?P<no>\d+)\s*조(?:\s*의\s*(?P<sub>\d+))?"
)


def _article(m) -> str:
    return f"제{m.group('no')}조" + (f"의{m.group('sub')}" if m.group("sub") else "")


def _evidence(text: str, m) -> str:
    """인용 근처 원문. 근거로 제시할 문장이 담기도록 넉넉히 뜬다.

    폭이 좁으면 "익증권 매매계약 취소 관련"처럼 법리가 없는 조각만 남는다.
    이 스니펫은 관리자 감사 화면에 보이고, 코퍼스 밖 법령의 경우 답변 근거로도
    쓰이므로 문장이 온전히 들어갈 만큼은 있어야 한다.
    """
    start = max(0, m.start() - 90)
    return text[start : m.end() + 130].replace("\n", " ").strip()


def extract_law_citations(text: str) -> list[tuple[str, str]]:
    """코퍼스 보유 법령 인용 → [(law_chunk_id, evidence)]. 중복 제거, 원문 순서 유지."""
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _CITE.finditer(text):
        law_name = None
        for pattern, name in _ALIAS:
            if re.fullmatch(pattern, m.group("alias")):
                law_name = name
                break
        if law_name is None:
            continue
        suffix = m.group("suffix")
        if suffix == "시행령":
            law_name = law_name + " 시행령"
        elif suffix == "감독규정":
            law_name = "금융소비자 보호에 관한 감독규정"
        chunk_id = f"{law_name}-{_article(m)}"
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        results.append((chunk_id, _evidence(text, m)))
    return results


def extract_stub_citations(text: str) -> list[tuple[str, str, str, str]]:
    """코퍼스 밖 법령 인용 → [(stub_id, law_name, article, evidence)].

    인용 자체는 결정례 원문의 사실이며, 노드는 law.go.kr 딥링크로 검증 가능하다.
    """
    results: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for m in _STUB_CITE.finditer(text):
        law_name = None
        for pattern, name in STUB_LAWS:
            if re.fullmatch(pattern, m.group("alias")):
                law_name = name
                break
        if law_name is None:
            continue
        if m.group("suffix") == "시행령":
            law_name = law_name + " 시행령"
        article = _article(m)
        stub_id = f"ref:{law_name}-{article}"
        if stub_id in seen:
            continue
        seen.add(stub_id)
        results.append((stub_id, law_name, article, _evidence(text, m)))
    return results


def stub_law_url(law_name: str, article: str) -> str:
    """국가법령정보센터 조문 딥링크 (예: law.go.kr/법령/민법/제109조)."""
    return f"https://www.law.go.kr/법령/{law_name}/{article}"
