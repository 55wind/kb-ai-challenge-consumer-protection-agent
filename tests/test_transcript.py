from app.agent import transcript as tr
from app.agent.answerer import Citation

RAW = """[00:10] 직원: 이 펀드는 원금 보장이라 걱정 없으세요
[00:25] 고객: 위험한 건 아니죠?
[00:31] 직원: 무조건 수익 나는 상품입니다"""


class StubLLM:
    def generate(self, *a, **k):
        return '{"level": "red", "reason": "원금보장 단정"}'


_CITATION = Citation(
    record_id="law:1", title="금융소비자보호법 제19조", source="국가법령정보센터",
    source_url="https://law.go.kr", quote="설명의무",
)


def _with_law(monkeypatch):
    """근거 조문이 검색되는 상황 (정상 경로)."""
    monkeypatch.setattr(tr, "_law_text_and_citation", lambda rule: ("조문 원문", _CITATION))
    monkeypatch.setattr(tr, "_related_cases", lambda q: [])


def _without_law(monkeypatch):
    """근거 조문을 찾지 못한 상황 (H-3 경로)."""
    monkeypatch.setattr(tr, "_law_text_and_citation", lambda rule: ("", None))
    monkeypatch.setattr(tr, "_related_cases", lambda q: [])


def test_parse_with_timestamp():
    utts, _ = tr.parse_transcript(RAW)
    assert len(utts) == 3
    assert utts[0].timestamp == "[00:10]" and utts[0].is_staff
    assert not utts[1].is_staff
    assert utts[2].seq == 3


def test_parse_without_timestamp():
    utts, _ = tr.parse_transcript("직원: 안내드립니다\n고객: 네")
    assert utts[0].timestamp == "" and len(utts) == 2


def test_parse_skips_non_dialog_lines():
    utts, _ = tr.parse_transcript("=== 녹취 시작 ===\n직원: 안내드립니다\n(잡음)\n고객: 네")
    assert len(utts) == 2


def test_only_staff_judged(monkeypatch):
    judged = []
    monkeypatch.setattr(
        tr, "_judge", lambda c, r, lt, llm: (judged.append(c), ("red", "사유"))[1]
    )
    _with_law(monkeypatch)
    timeline = tr.mine_evidence(RAW, StubLLM())
    assert judged and all("위험한 건" not in c for c in judged)
    assert timeline.summary.red == len(timeline.items) >= 1
    assert timeline.summary.total == 2  # 직원 발화 수만 집계
    assert timeline.scope == "staff"


def test_case_link_skipped_on_no_evidence(monkeypatch):
    monkeypatch.setattr(tr, "_judge", lambda c, r, lt, llm: ("red", "사유"))
    _with_law(monkeypatch)
    timeline = tr.mine_evidence(RAW, StubLLM())
    assert timeline.items and all(i.related_cases == [] for i in timeline.items)


def test_benign_staff_utterance_no_llm(monkeypatch):
    calls = []

    class Counting:
        def generate(self, *a, **k):
            calls.append(1)
            return '{"level": "red", "reason": "r"}'

    _with_law(monkeypatch)
    timeline = tr.mine_evidence("직원: 영업시간은 9시부터 4시까지입니다", Counting())
    assert calls == [] and timeline.items == []


# ── H-3: 근거 조문 없이는 위반 판정을 내지 않는다 ──────────────────
def test_no_law_yields_check_not_verdict(monkeypatch):
    calls = []

    class Counting:
        def generate(self, *a, **k):
            calls.append(1)
            return '{"level": "red", "reason": "r"}'

    _without_law(monkeypatch)
    timeline = tr.mine_evidence(RAW, Counting())
    assert calls == []                                  # LLM 판정 자체를 하지 않는다
    assert timeline.items and all(i.level == "check" for i in timeline.items)
    assert timeline.summary.red == 0 and timeline.summary.check >= 1
    assert all(i.law_citation is None for i in timeline.items)


# ── H-4: 화자 역할 고정 해제 ────────────────────────────────────────
AB_RAW = """[00:03] B: 이 상품 원금 보장되나요?
[00:07] A: 네 원금 보장이라 걱정 없으세요
[00:12] A: 무조건 수익 나는 상품입니다"""


def test_ab_labels_are_not_assumed_to_be_staff():
    """A가 직원이라는 가정을 제거 — 역할은 'unknown'이어야 한다."""
    utts, _ = tr.parse_transcript(AB_RAW)
    assert [u.role for u in utts] == ["unknown", "unknown", "unknown"]
    assert not any(u.is_staff for u in utts)


def test_unknown_roles_scan_all_utterances(monkeypatch):
    """역할 미확정이면 절반을 버리지 않고 전체를 검사하고, 그 사실을 알린다."""
    _with_law(monkeypatch)
    monkeypatch.setattr(tr, "_judge", lambda c, r, lt, llm: ("red", "사유"))
    timeline = tr.mine_evidence(AB_RAW, StubLLM())
    assert timeline.scope == "all"
    assert "모든 발언을 검사" in timeline.notice
    assert timeline.summary.red >= 1          # 예전에는 A만 보고 B를 버렸다


def test_korean_labels_beyond_whitelist_parse():
    """'창구직원'·'담당자'·실명 라벨도 조용히 버려지지 않는다."""
    utts, _ = tr.parse_transcript(
        "창구직원: 원금 보장입니다\n담당자: 확인해 드릴게요\n홍길동: 알겠습니다"
    )
    assert len(utts) == 3
    assert utts[0].role == "staff" and utts[1].role == "staff"
    assert utts[2].role == "unknown"


def test_named_speaker_is_not_dropped_when_staff_label_exists(monkeypatch):
    """실명 라벨 판매직원의 위반 발언이 조용히 사라지면 안 된다.

    '직원' 라벨이 하나라도 있으면 나머지를 버리던 동작이 H-4의 핵심 실패 모드였다.
    검사에서 빼는 것은 '고객'으로 확정된 발언뿐이다.
    """
    _with_law(monkeypatch)
    monkeypatch.setattr(tr, "_judge", lambda c, r, lt, llm: ("red", "사유"))
    raw = (
        "직원: 안녕하세요 어떤 상품 찾으세요\n"
        "김철수: 이건 무조건 원금 보장됩니다 손실 없어요\n"
        "고객: 정말 안전한가요"
    )
    timeline = tr.mine_evidence(raw, StubLLM())
    utterances = [i.utterance for i in timeline.items]
    assert any("무조건 원금 보장" in u for u in utterances), "실명 화자의 위반 발언이 누락됨"
    assert timeline.scope == "mixed"
    assert "확정하지 못해" in timeline.notice


def test_customer_utterances_still_excluded(monkeypatch):
    """확정된 고객 발언은 여전히 검사 대상이 아니다."""
    _with_law(monkeypatch)
    monkeypatch.setattr(tr, "_judge", lambda c, r, lt, llm: ("red", "사유"))
    timeline = tr.mine_evidence(
        "직원: 원금 보장입니다\n고객: 무조건 수익 나는 거 맞죠", StubLLM()
    )
    assert all("무조건 수익" not in i.utterance for i in timeline.items)
    assert timeline.scope == "staff" and timeline.notice == ""


def test_parse_failure_is_not_reported_as_safe(monkeypatch):
    """형식 미인식은 '위험 0건'이 아니라 '검사하지 못함'으로 구분된다."""
    _with_law(monkeypatch)
    timeline = tr.mine_evidence("오늘 상담 내용을 그냥 줄글로 적었습니다", StubLLM())
    assert timeline.parsed == 0
    assert timeline.scope == "none"
    assert "인식하지 못했" in timeline.notice
    assert timeline.summary.red == 0


def test_unparsed_lines_are_counted():
    """라벨처럼 생겼는데 인식 못 한 줄은 앞 화자에 합치지 않고 보고한다.

    합쳐버리면 남의 말이 앞 화자 발언으로 기록된다 — 증거로 쓰이는 데이터라 위험하다.
    """
    text = "직원: 안내드립니다\n" + "이것은화자라벨이라기에는너무긴문장입니다: 내용\n"
    utts, unread = tr.parse_transcript(text)
    assert len(utts) == 1
    assert "내용" not in utts[0].text          # 앞 발화에 섞이지 않는다
    assert unread == 1 and tr.count_unparsed(text) == 1


def test_wrapped_utterance_is_joined(monkeypatch):
    """줄바꿈으로 이어진 발언이 사라지면 안 된다.

    콜론이 없어 예전 집계로는 잡히지도 않았다 — 위반 문구가 unparsed=0인 채 증발했다.
    """
    _with_law(monkeypatch)
    monkeypatch.setattr(tr, "_judge", lambda c, r, lt, llm: ("red", "사유"))
    text = "직원: 이 상품은 안전하고\n무조건 원금 보장됩니다"
    utts, unread = tr.parse_transcript(text)
    assert len(utts) == 1 and unread == 0
    assert "무조건 원금 보장" in utts[0].text
    timeline = tr.mine_evidence(text, StubLLM())
    assert timeline.items, "이어진 줄의 위반 문구가 검사되지 않았다"


def test_structural_lines_not_counted_as_unread():
    """구분선·타임스탬프 표시를 미파싱으로 부풀리지 않는다."""
    text = "=== 녹취 시작 ===\n[00:10]\n직원: 안내드립니다\n(잡음)"
    utts, unread = tr.parse_transcript(text)
    assert len(utts) == 1 and unread == 0
