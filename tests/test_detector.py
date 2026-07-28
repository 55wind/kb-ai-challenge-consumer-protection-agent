import pytest

from app.agent.detector import detect_input_kind


# ── 녹취록 오분류 회귀 ────────────────────────────────────────────
# 판별이 느슨해지면 '라벨: 값' 형태의 문서·문자가 녹취 경로로 새고,
# 콜론 뒤 값만 발화로 남아 조항 검사가 무력화된다(문자 신고 안내도 사라진다).
NOT_TRANSCRIPT = {
    "상품설명서 항목": "상품명: KB주가연계펀드\n가입금액: 1000만원\n만기: 3년\n원금보장: 비보장",
    "약관 조문": "제1조: 목적\n제2조: 정의\n제3조: 적용범위",
    "이메일 머리말": "보낸사람: kb@kbstar.com\n받는사람: hong@test.com\n제목: 상품 안내",
    "메모형 문서": "참고: 설명 못 들었음\n비고: 녹취 있음\n출처: 창구 상담",
}


@pytest.mark.parametrize("name,text", list(NOT_TRANSCRIPT.items()))
def test_colon_lines_are_not_transcripts(name, text):
    assert detect_input_kind(text) != "transcript", name


def test_smishing_block_still_routes_to_sms():
    """스미싱 문자에 '확인:', '문의:' 줄이 있어도 문자 경로를 유지해야
    118/112/1332 신고 안내가 붙는다."""
    text = (
        "[국민은행] 고객님 계좌가 정지되었습니다\n"
        "확인: https://kb-check.com\n"
        "문의: 1588-9999\n\n"
        "[Web발신] (광고) 저금리 대출 안내\n"
        "수신거부: 080-000-0000"
    )
    assert detect_input_kind(text) == "sms_batch"

TRANSCRIPT = "직원: 이 상품은 안전합니다\n고객: 원금 보장되나요?\n직원: 네 걱정 마세요"
SMS_BATCH = "[Web발신] 대출 승인! http://bit.ly/x 무료수신거부\n\n(광고) 저금리 특별 한도 010-1234-5678"
DOCUMENT = "제1조 (목적) 이 약관은 여신거래에 관한 기본 사항을 정한다.\n제2조 (정의) ..."


def test_detects_transcript():
    assert detect_input_kind(TRANSCRIPT) == "transcript"


def test_detects_sms_batch():
    assert detect_input_kind(SMS_BATCH) == "sms_batch"


def test_document_fallback():
    assert detect_input_kind(DOCUMENT) == "document"


def test_single_speaker_label_is_document():
    assert detect_input_kind("직원: 이라고 적힌 안내문입니다. 상품 설명서 본문...") == "document"


def test_timestamp_with_speaker_is_transcript():
    """타임스탬프 + 화자 라벨이면 라벨이 실명이어도 녹취록이다."""
    assert detect_input_kind(
        "[00:12] 홍길동: 가입 권유드립니다\n[00:45] 김영희: 원금은 보장되나요"
    ) == "transcript"


def test_timestamp_without_speaker_is_document():
    """타임스탬프만 있고 화자 구분이 없으면 문서로 보낸다.

    이 형식은 parse_transcript가 발화를 하나도 못 뽑아 '형식을 인식하지 못했습니다'로
    끝난다. 문서 경로로 보내면 줄 전체가 조항으로 검사되므로 위반을 실제로 잡는다 —
    '타임스탬프 2개면 무조건 녹취록'이 위험 0건 오표시를 낳던 경로를 없앤 것이다.
    """
    assert detect_input_kind("[00:12] 가입 권유드립니다\n[00:45] 원금은 보장됩니다") == "document"


def test_single_block_with_url_is_document():
    assert detect_input_kind("자세한 내용은 https://fss.or.kr 참고하세요. 약관 전문입니다.") == "document"
