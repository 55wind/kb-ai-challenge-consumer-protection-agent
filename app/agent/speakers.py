"""녹취록 화자 규칙 — 판별(detector)과 파싱(transcript)이 공유하는 단일 출처.

두 모듈이 각자 정규식을 들고 있다가 한쪽만 고쳐 어긋난 적이 있다. 판별이 느슨해지면
상품설명서·문자함이 녹취록으로 새고, 그러면 조항 텍스트가 "라벨: 값"으로 잘려 룰 검사가
무력화된다. 그래서 규칙을 여기 한 곳에 둔다.

판별과 파싱은 요구 강도가 다르다.
- 판별(detector): **보수적**이어야 한다. 녹취록이라고 확신할 신호(타임스탬프, 역할 키워드,
  A/B 단문자 라벨)가 있을 때만 녹취 경로로 보낸다. 놓치면 문서 경로로 가는데, 문서 경로는
  줄 전체를 조항으로 검사하므로 위반 문구를 여전히 잡는다 — 안전한 실패 방향이다.
- 파싱(transcript): 녹취 경로로 확정된 뒤이므로 **관대**해도 된다. 실명·직책 등 어떤 라벨도
  받아 조용히 버리는 줄이 없게 한다.
"""
import re

# 역할을 '확정'할 수 있는 키워드. 추정하지 않는다 — A/B·실명은 unknown으로 남긴다.
STAFF_WORDS = (
    "직원", "상담사", "은행원", "설계사", "창구", "담당자", "매니저", "지점장",
    "텔러", "판매인", "컨설턴트", "어드바이저", "팀장", "대리", "과장", "차장", "부장",
    "PB", "FC", "FP",
)
CUSTOMER_WORDS = (
    "고객", "소비자", "본인", "가입자", "피해자", "민원인", "신청인",
    "어머니", "아버지", "아들", "딸", "나", "저",
)

# 라벨처럼 생겼지만 화자가 아닌 것 (문서 머리말·URL 스킴 등)
NOT_SPEAKER = {"http", "https", "참고", "비고", "메모", "출처", "문의", "확인", "안내", "제목"}

_TS = r"[\[(]\d{1,2}:\d{2}(?::\d{2})?[\])]"

# ── 파싱용(관대): 타임스탬프 + 12자 이내 임의 라벨 ──
LINE = re.compile(
    rf"^\s*(?P<ts>{_TS})?\s*"
    r"(?P<speaker>[^\s:：\[\]()][^:：]{0,11}?)\s*[:：]\s*(?P<text>\S.*)$"
)

# ── 판별용(보수적) ──
# (1) 타임스탬프가 앞에 붙은 줄은 라벨이 무엇이든 녹취록 줄로 본다
TS_LINE = re.compile(rf"(?m)^\s*{_TS}\s*[^\s:：\[\]()][^:：\n]{{0,11}}?\s*[:：]\s*\S")
# (2) 타임스탬프가 없으면 역할 키워드나 A/B 같은 단문자 라벨만 인정한다.
#     '상품명:', '문의:', '제1조:' 같은 문서·문자 라인이 새지 않게 하는 핵심 제약이다.
_ROLE_ALT = "|".join(sorted((*STAFF_WORDS, *CUSTOMER_WORDS), key=len, reverse=True))
ROLE_LINE = re.compile(rf"(?m)^\s*(?:{_ROLE_ALT})\w*\s*[:：]\s*\S")
AB_LINE = re.compile(r"(?m)^\s*[A-Z]\s*[:：]\s*\S")


def classify_speaker(label: str) -> str:
    """라벨 → 'staff' | 'customer' | 'unknown'. 추정하지 않고 확정만 한다."""
    norm = label.strip()
    upper = norm.upper()
    if any(w in norm or w in upper for w in STAFF_WORDS):
        return "staff"
    if any(w in norm for w in CUSTOMER_WORDS):
        return "customer"
    return "unknown"


def transcript_line_count(text: str) -> int:
    """녹취록이라고 볼 만한 줄 수 (판별 전용)."""
    return len(TS_LINE.findall(text)) + len(ROLE_LINE.findall(text)) + len(AB_LINE.findall(text))
