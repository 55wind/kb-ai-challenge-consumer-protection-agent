"""데이터 수집 절차 문서화 스크립트.

fss.or.kr는 일반 HTTP 클라이언트를 차단하므로(WAF) 수집은 실브라우저
(gstack browse 데몬 또는 수동 다운로드)로 수행했다. 이 스크립트는 수집
절차를 문서화하고 수집 상태를 점검한다.

수집 절차 (2026-07-23 수행):
1. 분쟁조정결정례 목록 크롤: https://www.fss.or.kr/fss/bbs/B0000390/list.do
   ?menuNo=&viewType=BODY&pageIndex=N (N=1..10) 에서 view.do?nttId=* 링크 수집
2. 각 상세 페이지의 fileDown.do 첨부(HWP)를 data/raw/cases/case_{nttId}_{n}.hwp로
   다운로드, manifest.tsv에 제목·등록일·URL 기록
3. 법령: law.go.kr에서 금소법/시행령/감독규정 본문 iframe(lsInfoP.do) 텍스트를
   data/raw/laws/*.txt + *.meta.json으로 저장
4. 절차: fss.or.kr 안내 페이지(menuNo=200644,201179,200518,200521) 본문을
   data/raw/procedures/*.json으로 구조화

이후: python scripts/build_corpus.py {cases,laws,procedures}
검증: python scripts/validate_corpus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings


def main() -> int:
    raw = settings.DATA_DIR / "raw"
    ok = True
    for sub, pattern, minimum in (
        ("cases", "*.hwp", 30),
        ("laws", "*.txt", 3),
        ("procedures", "*.json", 4),
    ):
        found = len(list((raw / sub).glob(pattern)))
        status = "OK" if found >= minimum else "부족"
        if found < minimum:
            ok = False
        print(f"[{status}] data/raw/{sub}: {found}개 (최소 {minimum})")
    if not ok:
        print("→ 모듈 독스트링의 수집 절차를 다시 수행하세요.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
