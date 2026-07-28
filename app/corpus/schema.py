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


class ProcedureDoc(_SourcedRecord):
    title: str
    steps: list[str]
    required_documents: list[str]
    text: str


class ProductChunk(_SourcedRecord):
    """KB 상품 약관·상품설명서 청크 (조항 또는 단락 단위)."""

    product_name: str
    doc_type: Literal["약관", "특약", "상품설명서"]
    section: str
    text: str
