"""근거 인용 답변 생성. 인용 없는 답변은 UngroundedAnswer로 거부."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.llm.ollama_client import OllamaClient
from app.search.retriever import Hit


class UngroundedAnswer(Exception):
    """생성 답변이 근거 문서를 하나도 인용하지 않음."""


@dataclass
class Citation:
    record_id: str
    title: str
    source: str
    source_url: str
    quote: str


@dataclass
class GlossaryItem:
    term: str
    definition: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    glossary: list[GlossaryItem] = field(default_factory=list)


_glossary_cache: dict[str, str] | None = None


def _load_glossary() -> dict[str, str]:
    global _glossary_cache
    if _glossary_cache is None:
        path = settings.CORPUS_DIR / "glossary.json"
        _glossary_cache = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
    return _glossary_cache


def _doc_title(record) -> str:
    if getattr(record, "product_name", None):
        return f"{record.product_name} {record.doc_type} {record.section}"
    return getattr(record, "title", None) or getattr(record, "article", None) or record.id


def _doc_text(record) -> str:
    for attr in ("text", "reasoning", "facts"):
        if v := getattr(record, attr, None):
            # 절차 문서는 본문이 개요뿐이다. 단계·필요서류를 함께 넘기지 않으면
            # "무슨 서류가 필요한가요"에 근거를 대고도 답을 못 한다.
            steps = getattr(record, "steps", None)
            docs = getattr(record, "required_documents", None)
            if steps or docs:
                extra = []
                if steps:
                    extra.append("처리 절차:\n" + "\n".join(steps))
                if docs:
                    extra.append("필요 서류:\n" + "\n".join(f"- {d}" for d in docs))
                return v + "\n\n" + "\n\n".join(extra)
            return v
    return ""


# 사전에 있지만 일상어와 겹쳐 오탐을 내는 항목.
# 부분 문자열 매칭이라 "불공정 조항 검사"의 '검사', "상품 모집"의 '모집'까지
# 금융용어로 잡혀 엉뚱한 풀이가 붙었다. 뜻이 겹치는 순간 설명이 오히려 방해가 된다.
_AMBIGUOUS_TERMS = frozenset({
    "검사", "매출", "모집", "중요도", "기여도", "옵션", "신탁", "사모", "스왑",
})

# 풀이가 필요한 어려운 말인지의 하한. 두 글자짜리는 대개 일상어와 겹친다.
_MIN_TERM_LEN = 3

# 짧아도 소비자가 모르는 약어·외래어는 남긴다
_KEEP_SHORT = frozenset({"CMA", "델타", "파밍", "피싱", "부도율", "손해율", "보험료"})


def find_glossary_terms(text: str) -> list[GlossaryItem]:
    """본문에 나온 어려운 금융용어의 쉬운 풀이.

    부분 문자열 매칭이라 짧은 용어일수록 오탐이 난다. 일상어와 겹치는 항목과
    두 글자 용어를 걸러, 풀이가 실제로 도움이 되는 것만 남긴다.
    """
    return [
        GlossaryItem(term=t, definition=d)
        for t, d in _load_glossary().items()
        if t in text
        and t not in _AMBIGUOUS_TERMS
        and (len(t) >= _MIN_TERM_LEN or t in _KEEP_SHORT)
    ]


SYSTEM_PROMPT = (
    "너는 금융소비자 보호 상담 도우미다. 규칙:\n"
    "1. 반드시 제공된 근거 문서의 내용만으로 답한다. 근거에 없는 내용은 추측하지 않는다.\n"
    "2. 근거를 사용한 각 주장 뒤에 해당 문서 번호를 [1] 형태로 표기한다.\n"
    "3. 근거가 부족하면 '근거 부족'이라고 명시한다.\n"
    "4. 쉬운 한국어로, 소비자의 눈높이에서 설명한다.\n"
    "5. 법률 자문이 아닌 정보 제공임을 전제로 한다."
)


def _build_prompt(query: str, hits: list[Hit]) -> str:
    docs = "\n\n".join(
        f"[{i + 1}] {_doc_title(h.record)} (출처: {h.record.source})\n{_doc_text(h.record)[:1200]}"
        for i, h in enumerate(hits)
    )
    return f"근거 문서:\n{docs}\n\n소비자 질문: {query}\n\n답변:"


def _validate(text: str, hits: list[Hit]) -> Answer:
    """생성문에 근거 인용이 있는지 검사해 Answer로 확정한다.

    스트리밍이든 아니든 이 검사를 통과해야 답변이 된다 — 화면에 글자가 먼저
    흘렀더라도 인용이 없으면 답변으로 확정하지 않는다.
    """
    used_nums = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    valid = [n for n in used_nums if 1 <= n <= len(hits)]
    if not valid:
        raise UngroundedAnswer("답변에 근거 인용이 없습니다")

    citations = [
        Citation(
            record_id=hits[n - 1].record_id,
            title=_doc_title(hits[n - 1].record),
            source=hits[n - 1].record.source,
            source_url=hits[n - 1].record.source_url,
            quote=_doc_text(hits[n - 1].record)[:200],
        )
        for n in sorted(valid)
    ]
    return Answer(text=text, citations=citations, glossary=find_glossary_terms(text))


def answer(query: str, hits: list[Hit], llm: OllamaClient) -> Answer:
    text = llm.generate(SYSTEM_PROMPT, _build_prompt(query, hits), timeout=180).strip()
    return _validate(text, hits)


def answer_streaming(query: str, hits: list[Hit], llm: OllamaClient):
    """토큰을 흘리며 생성하고, 끝나면 검증한다.

    CPU 추론은 답변 하나에 수십 초가 걸린다. 완성될 때까지 화면이 비어 있으면
    사용자는 멈춘 줄 안다. 그래서 조각을 먼저 보내되 **확정은 검증 후**에 한다.
    인용이 없으면 UngroundedAnswer를 올리고, 화면은 잠정 표시를 버린다.

    yield: ("delta", 토큰) 여러 번 → 마지막에 ("answer", Answer)
    """
    prompt = _build_prompt(query, hits)
    stream = getattr(llm, "stream", None)
    if stream is None:
        # 스트리밍을 지원하지 않는 LLM(테스트 스텁·대체 백엔드)이면 한 번에 받는다.
        # 스트리밍은 대기 체감을 줄이는 장치이지 답변의 조건이 아니다.
        yield ("answer", _validate(llm.generate(SYSTEM_PROMPT, prompt, timeout=180).strip(), hits))
        return
    buf: list[str] = []
    for token in stream(SYSTEM_PROMPT, prompt, timeout=300):
        buf.append(token)
        yield ("delta", token)
    yield ("answer", _validate("".join(buf).strip(), hits))


_RETRY_HINT = "\n(반드시 근거 번호 [n]을 표기해 다시 답하세요)"


def answer_events(query: str, hits: list[Hit], llm: OllamaClient):
    """오케스트레이터가 쓰는 단일 진입점.

    스트리밍이든 일괄이든 **인용 검증과 1회 재시도**를 똑같이 거친다.
    진입점이 갈리면 한쪽만 보장이 빠지는 사고가 난다.

    yield: ("delta", 토큰)* → ("answer", Answer)
    """
    if not settings.ANSWER_STREAMING:
        yield ("answer", answer_with_retry(query, hits, llm))
        return
    try:
        yield from answer_streaming(query, hits, llm)
    except UngroundedAnswer:
        # 첫 시도의 잠정 표시를 버리게 알리고, 인용을 요구해 다시 생성한다
        yield ("discard", None)
        yield from answer_streaming(query + _RETRY_HINT, hits, llm)


def answer_with_retry(query: str, hits: list[Hit], llm: OllamaClient) -> Answer:
    """인용 누락 시 1회 재시도, 재실패면 UngroundedAnswer 전파."""
    try:
        return answer(query, hits, llm)
    except UngroundedAnswer:
        return answer(query + "\n(반드시 근거 번호 [n]을 표기해 다시 답하세요)", hits, llm)
