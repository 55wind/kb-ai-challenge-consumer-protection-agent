"""코퍼스 레코드 스키마.

데이터 원칙(설계서 §4): 모든 레코드는 실제 출처(source, source_url)가 있어야 한다.
출처 없는 레코드는 CorpusValidationError로 거부한다. 테스트 픽스처는
source에 '[TEST-FIXTURE]'를 표기해야 하며, 실서비스 코퍼스에 섞이면
validate_corpus.py가 잡아낸다.
"""
from typing import Literal

from pydantic import BaseModel


class CorpusValidationError(Exception):
    pass


class _SourcedRecord(BaseModel):
    id: str
    source: str
    source_url: str

    @classmethod
    def validated(cls, **kwargs):
        record = cls(**kwargs)
        if not record.source.strip() or not record.source_url.strip():
            raise CorpusValidationError(
                f"{cls.__name__} {record.id!r}: source/source_url은 비울 수 없습니다. "
                "실제 출처가 없는 데이터는 코퍼스에 넣을 수 없습니다."
            )
        return record


class LawChunk(_SourcedRecord):
    law_name: str
    article: str
    text: str


class CaseRecord(_SourcedRecord):
    title: str
    facts: str
    issues: str
    reasoning: str
    decision: Literal["인용", "일부인용", "기각"]
    decision_date: str
    category: str
    # 그 category를 누가 정했는지. 이 구분이 없으면 우리 추정이 기관 분류인 척한다.
    #
    #   "금융감독원 사례집"  금감원 「금융분쟁조정 사례집」의 사례 목차와 제목이
    #                      일치해 그쪽 분류를 옮겨 온 것. 기관이 붙인 라벨이다.
    #   "키워드 규칙(추정)"  사례집에 없어 build_corpus.py의 키워드 규칙이 정한 것.
    #                      금감원 라벨과 대조 가능한 17건에서 **오류율 18%**였다.
    #
    # 화면·문서에서 정확도를 말할 때 이 값을 함께 봐야 한다.
    category_source: Literal["금융감독원 사례집", "키워드 규칙(추정)"] = "키워드 규칙(추정)"


class ProcedureDoc(_SourcedRecord):
    title: str
    steps: list[str]
    required_documents: list[str]
    text: str


class InterpretationDoc(_SourcedRecord):
    """감독당국(금융위·금감원)이 낸 금소법 유권해석·설명자료.

    조문과 분리해서 보관한다. 조문은 법이고 이것은 해석이라 **효력이 다르다** —
    인용 화면에서 같은 무게로 보이면 사용자가 오해한다. doc_type이 그 구분을
    유지하고, 화면은 이 값으로 출처 표기를 달리한다.
    """

    # FAQ  업권 실무자가 물은 것에 감독당국이 답한 문답 (금융위 금소법 FAQ 등)
    # 해설  조문 해설서 (금감원 설명자료) — 문답 구조가 아니다
    # 안내  **소비자에게** 설명하려고 쓴 문답 (금감원 금융소비자용 안내자료).
    #       질문이 구어체라 다른 코퍼스가 못 가진 문체를 담는다 —
    #       "금융상품 거래를 되돌릴 수 있나요(청약철회권)?" 같은 것들.
    # 법령해석  금융위가 **개별 질의에 회신한 유권해석** (금융규제·법령해석포털).
    #       FAQ와 다른 점은 사안이 특정되어 있다는 것이다 — 조문의 뜻이 갈리는
    #       구체적 자리에 당국이 무엇이라 답했는지가 담긴다. 질의요지/회답/이유
    #       세 구간으로 오고, '이유'가 조문을 직접 인용한다.
    #       효력은 FAQ와 같다(법이 아니라 해석) — doc_type을 나누는 것은
    #       화면에서 출처를 다르게 적기 위한 것이고, 무게를 올리는 뜻이 아니다.
    doc_type: Literal["FAQ", "해설", "안내", "법령해석"]
    topic: str  # 문서 안의 章 제목 (예: "Ⅲ. 청약철회권")
    title: str  # FAQ면 질문, 해설이면 그 쪽의 논점
    text: str
    # 본문이 인용한 조문 "번호"만 담는다. 어느 법의 조문인지는 담기지 않는다 —
    # 한 답변이 금소법·시행령·감독규정·민법을 함께 인용하기 때문이다
    # (예: 연대보증 FAQ는 감독규정 §14와 민법 제437·439조를 같이 든다).
    #
    # **그래서 이 값으로 그래프 엣지를 만들면 안 된다.** `제437조`를 금소법
    # 제437조로 이어 붙이면 존재하지도 않는 조문에 엣지가 생긴다. 컨텍스트
    # 그래프는 전 엣지가 provenance를 갖는 것이 원칙이라, 근거가 모호한 연결은
    # 넣지 않는다. 지금 쓰임은 하나다 — 골든셋을 짤 때 정답 후보를 좁히는 것.
    cited_articles: list[str] = []


class ProductChunk(_SourcedRecord):
    """KB 상품 약관·상품설명서 청크 (조항 또는 단락 단위)."""

    product_name: str
    doc_type: Literal["약관", "특약", "상품설명서"]
    section: str
    text: str
