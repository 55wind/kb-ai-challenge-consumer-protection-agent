import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from app.config import settings
from app.corpus.schema import (
    CaseRecord,
    CorpusValidationError,
    LawChunk,
    ProcedureDoc,
    ProductChunk,
)

_MODELS = {
    "laws": LawChunk,
    "cases": CaseRecord,
    "procedures": ProcedureDoc,
    "kb_products": ProductChunk,
}

CorpusKind = Literal["laws", "cases", "procedures", "kb_products"]


def load_corpus(kind: CorpusKind, corpus_dir: Path | None = None) -> list:
    model = _MODELS[kind]
    path = (corpus_dir or settings.CORPUS_DIR) / f"{kind}.jsonl"
    if not path.exists():
        raise CorpusValidationError(f"코퍼스 파일이 없습니다: {path}")
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(model.validated(**json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as e:
                raise CorpusValidationError(f"{path.name}:{lineno} 파싱 실패: {e}") from e
    return records
