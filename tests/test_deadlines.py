from datetime import date

from app.agent import deadlines
from app.agent.deadlines import add_business_days, compute_deadlines, parse_event_date

TODAY = date(2026, 7, 23)


def test_parse_absolute():
    assert parse_event_date("2026년 5월 12일에 가입", TODAY) == date(2026, 5, 12)


def test_parse_month_only_defaults_day1():
    assert parse_event_date("2026년 5월", TODAY) == date(2026, 5, 1)


def test_parse_slash_future_rolls_back_year():
    assert parse_event_date("12/25에 있었던 일", TODAY) == date(2025, 12, 25)


def test_parse_relative_months():
    assert parse_event_date("3개월 전쯤", TODAY) == date(2026, 4, 23)


def test_parse_last_year():
    assert parse_event_date("작년 12월", TODAY) == date(2025, 12, 1)


def test_parse_failure():
    assert parse_event_date("예전에", TODAY) is None


def _no_law(monkeypatch):
    monkeypatch.setattr(deadlines, "_law_citation", lambda q: None)


def test_loan_urgent_boundary(monkeypatch):
    _no_law(monkeypatch)
    items = compute_deadlines("대출·금리", "2026년 7월 15일", today=TODAY)
    loan = next(i for i in items if "청약철회" in i.action)
    assert loan.urgency == "urgent"
    assert loan.computed_deadline == "2026-07-29"


def test_loan_expired(monkeypatch):
    _no_law(monkeypatch)
    items = compute_deadlines("대출·금리", "2026년 1월 5일", today=TODAY)
    assert next(i for i in items if "청약철회" in i.action).urgency == "expired"


def test_unknown_when_unparsable(monkeypatch):
    _no_law(monkeypatch)
    items = compute_deadlines("대출·금리", "예전에", today=TODAY)
    assert all(i.urgency in ("unknown", "urgent") for i in items)
    assert any("시기를 알려주시면" in i.urgency_note for i in items)


def test_phishing_immediate_first(monkeypatch):
    _no_law(monkeypatch)
    items = compute_deadlines("보이스피싱·전자금융사기", "오늘", today=TODAY)
    assert items[0].urgency == "urgent"
    assert "지급정지" in items[0].action
    assert any("분쟁조정" in i.action for i in items)  # 공통 룰 부착


def test_caveat_always_present(monkeypatch):
    _no_law(monkeypatch)
    items = compute_deadlines("불완전판매", "2026년 7월 1일", today=TODAY)
    assert all("1332" in i.urgency_note for i in items)


def test_mispanmae_investment_withdrawal_seven_days(monkeypatch):
    _no_law(monkeypatch)
    items = compute_deadlines("불완전판매", "2026년 7월 20일", today=TODAY)
    inv = next(i for i in items if "청약철회" in i.action)
    assert inv.computed_deadline == "2026-07-27"  # 7일 (제46조 원문 검증값)


# ── 영업일 산정 (C-3) ───────────────────────────────────────────────
# 통신사기피해환급법 시행령 제3조 ②는 "3영업일"이다. 역일 3일로 계산하면
# 금요일 신청 시 법정 기한과 이틀 어긋난다.
def test_add_business_days_skips_weekend():
    friday = date(2026, 7, 24)          # 금
    assert friday.weekday() == 4
    assert add_business_days(friday, 3) == date(2026, 7, 29)   # 월·화·수
    assert add_business_days(friday, 1) == date(2026, 7, 27)   # 다음 월요일


def test_add_business_days_from_weekend():
    saturday = date(2026, 7, 25)
    assert add_business_days(saturday, 1) == date(2026, 7, 27)


def test_phishing_written_filing_uses_business_days(monkeypatch):
    """금요일 전화 신청 → 서면 제출 기한은 역일 3일(7/27)이 아니라 영업일 3일(7/29)."""
    _no_law(monkeypatch)
    items = compute_deadlines("보이스피싱·전자금융사기", "2026년 7월 24일",
                              today=date(2026, 7, 24))
    written = next(i for i in items if "서면 제출" in i.action)
    assert written.computed_deadline == "2026-07-29"
    assert "주말 제외" in written.urgency_note


def test_no_business_day_caveat_when_not_computed(monkeypatch):
    """계산하지 못한 기한에 '주말 제외로 계산했습니다'를 붙이면 안 된다."""
    _no_law(monkeypatch)
    items = compute_deadlines("보이스피싱·전자금융사기", "언제인지 기억이 안 나요", today=TODAY)
    written = next(i for i in items if "서면 제출" in i.action)
    assert written.computed_deadline is None
    assert "주말 제외" not in written.urgency_note
    assert "시기를 알려주시면" in written.urgency_note


def test_calendar_day_rules_unchanged(monkeypatch):
    """역일 기준 룰(청약철회 등)은 영업일 계산의 영향을 받지 않는다."""
    _no_law(monkeypatch)
    items = compute_deadlines("대출·금리", "2026년 7월 15일", today=TODAY)
    loan = next(i for i in items if "청약철회" in i.action)
    assert loan.computed_deadline == "2026-07-29"
    assert "주말 제외" not in loan.urgency_note
