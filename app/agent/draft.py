"""분쟁조정 신청서 초안 생성: 템플릿 필드 채움 + sLLM 문장 다듬기."""
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.slots import Slots
from app.llm.ollama_client import OllamaClient

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "mediation_form.md"

# 유형별 증빙서류 (금감원 민원신청 안내 및 분쟁조정 절차 기반)
EVIDENCE_BY_CATEGORY: dict[str, list[str]] = {
    "불완전판매": [
        "상품 가입 계약서·약관 사본",
        "상품(투자)설명서, 가입 당시 안내자료",
        "판매 직원과의 대화 녹취·문자·카카오톡 캡처",
        "거래내역서(손실 확인용)",
    ],
    "보이스피싱·전자금융사기": [
        "경찰 신고 접수증(사건사고사실확인원)",
        "이체·출금 거래내역서",
        "사기 문자·통화 기록 캡처",
        "지급정지 신청 내역",
    ],
    "대출·금리": [
        "대출 계약서·상환 내역",
        "금리 안내·변경 통지 자료",
        "금리인하요구 신청·회신 내역",
    ],
    "부당권유·불공정영업": [
        "가입 권유 당시 녹취·문자 기록",
        "계약서·상품설명서",
        "타 상품 가입 강요 정황 자료",
    ],
}
_DEFAULT_EVIDENCE = [
    "관련 계약서·약관 사본",
    "거래내역서",
    "금융회사와 주고받은 문자·통화 기록",
    "신분증 사본 (제출 시)",
]


@dataclass
class Draft:
    markdown: str
    # 유형별로 준비해야 할 증빙 목록 (전체). 이름이 missing_evidence였을 때는
    # "빠진 증빙"을 뜻하는 것처럼 보였지만 실제로는 늘 전체 목록이었다.
    evidence_checklist: list[str] = field(default_factory=list)
    # 초안에 "(작성 필요)"로 남은 항목 — 사용자가 직접 채워야 제출할 수 있다
    missing_fields: list[str] = field(default_factory=list)


def _evidence_appendix(timeline) -> str:
    rows = "\n".join(
        f"| {i.timestamp or '-'} | \"{i.utterance[:60]}\" | {i.rule} | "
        f"{i.law_citation.title if i.law_citation else '-'} |"
        for i in timeline.items
    )
    return (
        "\n## 별첨: 상담 녹취 위반 의심 발언 타임라인\n\n"
        "| 시점 | 발언 | 위반 의심 유형 | 근거 조문 |\n"
        "|---|---|---|---|\n"
        f"{rows}\n"
    )


def build_draft(slots: Slots, llm: OllamaClient, category: str = "기타",
                respondent: str = "", evidence_timeline=None) -> Draft:
    evidence = list(EVIDENCE_BY_CATEGORY.get(category, _DEFAULT_EVIDENCE))
    filled = {k: slots.filled.get(k, "(작성 필요)") for k in
              ("who", "when", "product", "incident", "demand")}

    # 경위·취지는 sLLM으로 문어체 다듬기 (실패 시 원문 유지)
    if slots.filled.get("incident"):
        polished = llm.generate(
            "다음 소비자 진술을 민원 신청서에 쓸 객관적 문어체 경위서로 2~4문장으로 다듬어라. "
            "사실만 쓰고 새로운 내용을 추가하지 마라.",
            slots.filled["incident"],
            timeout=120,
        ).strip()
        if polished:
            filled["incident"] = polished

    checklist = [f"- [ ] {e}" for e in evidence]
    if evidence_timeline and evidence_timeline.items:
        n = len(evidence_timeline.items)
        checklist.append(f"- [x] 상담 녹취록 및 위반 의심 발언 타임라인 (별첨, {n}건)")

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    markdown = template.format(
        who=filled["who"],
        respondent=respondent or "(금융회사명 작성 필요)",
        product=filled["product"],
        when=filled["when"],
        incident=filled["incident"],
        demand=filled["demand"],
        evidence_checklist="\n".join(checklist),
    )
    if evidence_timeline and evidence_timeline.items:
        markdown += _evidence_appendix(evidence_timeline)
    from app.agent.slots import SLOT_LABEL

    missing = [SLOT_LABEL.get(k, k) for k in SLOT_LABEL if k not in slots.filled]
    return Draft(markdown=markdown, evidence_checklist=evidence, missing_fields=missing)
