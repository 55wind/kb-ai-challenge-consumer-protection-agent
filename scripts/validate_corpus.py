"""코퍼스 검증: 3개 JSONL 전부 로드하고 출처 메타데이터를 강제한다.

실서비스 코퍼스에 [TEST-FIXTURE] 레코드가 섞여 있으면 실패한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 기본 코드페이지(cp949)에서 한글 출력이 깨지지 않도록 강제
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.corpus.loader import load_corpus
from app.corpus.schema import CorpusValidationError


def main() -> int:
    failed = False
    for kind in ("laws", "cases", "procedures", "kb_products"):
        try:
            records = load_corpus(kind)
        except CorpusValidationError as e:
            print(f"[FAIL] {kind}: {e}")
            failed = True
            continue
        fixtures = [r.id for r in records if "[TEST-FIXTURE]" in r.source]
        if fixtures:
            print(f"[FAIL] {kind}: 테스트 픽스처가 실코퍼스에 포함됨: {fixtures}")
            failed = True
            continue
        print(f"[OK] {kind}: {len(records)}건")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
