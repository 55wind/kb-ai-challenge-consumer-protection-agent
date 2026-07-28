"""검진 입력 3종 판별 — 결정론 휴리스틱, LLM 호출 없음 (spec §2)."""
import re
from typing import Literal

from app.agent.speakers import transcript_line_count

_SMS_MARKERS = re.compile(
    r"\[Web발신\]|\(광고\)|https?://|\d{2,3}-\d{3,4}-\d{4}|무료수신거부|수신거부"
)


def detect_input_kind(text: str) -> Literal["transcript", "sms_batch", "document"]:
    """녹취록 판별은 보수적으로 한다.

    화자 라벨이 확실할 때만 녹취 경로로 보낸다. 느슨하게 잡으면 '상품명: …' 같은
    문서나 '문의: 1588-9999' 같은 문자가 녹취록으로 새는데, 그러면 콜론 뒤 값만
    발화로 남아 조항 검사가 무력화되고 문자 신고 안내(118/112/1332)도 사라진다.
    반대로 놓친 녹취록은 문서 경로에서 줄 전체가 조항으로 검사되므로 위반은 잡힌다.
    """
    if transcript_line_count(text) >= 2:
        return "transcript"
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) >= 2 and _SMS_MARKERS.search(text):
        return "sms_batch"
    return "document"
