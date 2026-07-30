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


def test_illegal_contract_termination_is_offered_for_every_category():
    """위법계약해지는 유형과 무관하게 항상 안내해야 한다.

    금소법 제47조①이 적용 대상 위반을 열거한다 — 제17조③(적합성), 제18조②(적정성),
    제19조①③(설명의무), 제20조①(불공정영업), 제21조(부당권유). 이 위반들은
    보장성·투자성·예금성·대출성 **어느 상품에서도** 일어난다. 조문이 상품 유형을
    가리지 않으므로 이 권리도 유형에 묶여선 안 된다.

    예전에는 `불완전판매`와 `부당권유·불공정영업`에만 달려 있어서, 같은 설명의무
    위반이라도 분류가 `보험`이나 `대출·금리`로 가면 카드가 사라졌다. 분류 정확도가
    완전할 수 없으므로(금감원 라벨 기준 60%대) 이 결합이 곧 권리 누락이었다.
    """
    from app.agent.classifier import CATEGORIES
    from app.agent.deadlines import RULES

    for category in CATEGORIES:
        actions = [r.action for r in RULES.get(category, []) + RULES["_common"]]
        assert any("위법계약 해지" in a for a in actions), (
            f"'{category}'에 위법계약해지 안내가 없다 — 분류가 이 유형으로 틀리면 "
            f"제47조상 권리가 화면에서 사라진다"
        )


def test_withdrawal_period_matches_the_statute_per_product_type():
    """청약철회는 상품 유형별 기간이 제46조①과 맞아야 한다.

    제46조①은 유형마다 기간을 달리 정한다 — 1호 보장성 15일, 2호 투자성 7일,
    3호 대출성 14일. 이쪽은 조문이 유형을 구분하므로 유형에 묶는 것이 맞다.
    """
    from app.agent.deadlines import RULES

    expected = {"보험": ("보장성", 15), "불완전판매": ("투자성", 7),
                "투자상품": ("투자성", 7), "대출·금리": ("대출성", 14)}
    for category, (kind, days) in expected.items():
        hits = [r for r in RULES[category] if "청약철회" in r.action]
        assert hits, f"'{category}'에 청약철회 카드가 없다"
        rule = next(r for r in hits if kind in r.action)
        assert rule.days == days, f"{kind} 청약철회가 {rule.days}일 — 제46조는 {days}일"


def test_withdrawal_card_follows_product_type_not_complaint_category(monkeypatch):
    """청약철회는 민원 유형이 아니라 상품유형(금소법 제3조)으로 골라야 한다.

    제46조①이 상품유형별로 기간을 정한다 — 1호 보장성 15일, 2호 투자성 7일,
    3호 대출성 14일. 그런데 예전에는 `classifier.CATEGORIES` 10종으로 골랐다.
    그 10종은 상품 축과 위반 축을 섞어 놓아 배타적이 아니고(보험 불완전판매
    사건은 둘 다다) 금감원 라벨 기준 정확도가 62.5%였다. 유형을 잘못 집으면
    해당하는 청약철회 카드가 사라진다.

    상품유형 축 하나만 보면 정답 포함률이 96.0%다. 이 테스트는 유형이 틀려도
    상황 텍스트로 올바른 청약철회 카드가 붙는지 본다.
    """
    from app.agent import deadlines as dl
    from app.agent import product_type

    # 민원 유형은 '대출·금리'로 잘못 분류됐지만, 상황은 보험 사건이다.
    monkeypatch.setattr(product_type, "infer", lambda t, top=2: ["보장성 상품", "투자성 상품"])
    monkeypatch.setattr(dl, "_law_citation", lambda q: None)
    items = dl.compute_deadlines("대출·금리", "", situation_text="보험금을 안 준대요")
    actions = [i.action for i in items]

    assert any("보장성 상품 청약철회" in a for a in actions), (
        f"상품유형이 보장성인데 보장성 청약철회가 없다: {actions}")
    assert not any("대출성 상품 청약철회" in a for a in actions), (
        f"민원 유형(대출·금리)에서 온 청약철회가 남아 있다: {actions}")
    # 위법계약해지는 유형과 무관하게 항상 남아야 한다
    assert any("위법계약 해지" in a for a in actions)


def test_deadlines_survive_product_type_failure(monkeypatch, capsys):
    """상품유형 추정이 실패해도 기한 안내는 나가야 한다."""
    from app.agent import deadlines as dl
    from app.agent import product_type

    def boom(text, top=2):
        raise RuntimeError("임베딩 모델 없음")

    monkeypatch.setattr(product_type, "infer", boom)
    monkeypatch.setattr(dl, "_law_citation", lambda q: None)
    items = dl.compute_deadlines("불완전판매", "", situation_text="펀드 손실")
    assert items, "추정 실패로 기한 안내가 통째로 사라졌다"
    assert "상품유형 추정 실패" in capsys.readouterr().out


# ── 인접 유형 기한 함께 노출 ─────────────────────────────────────
def test_alt_category_deadlines_are_marked_tentative():
    """분류 2순위의 기한도 준다. 단, '해당될 수 있음'으로 구분된다.

    분류 recall@1은 금감원 라벨 기준 57%대인데 recall@2는 72%대다. 1순위만 쓰면
    그 차이만큼 법정 기한 카드가 사라지고, 카드가 사라지는 것은 보수적 방향이
    아니다. 반대로 확정 카드와 섞어 놓으면 화면이 사실을 과장한다.
    """
    items = compute_deadlines(
        "대출·금리", "2026년 7월 20일", today=TODAY,
        alt_category="보이스피싱·전자금융사기",
    )
    by_action = {i.action: i for i in items}
    urgent = next(a for a in by_action if "지급정지" in a)
    assert by_action[urgent].tentative, "2순위 카드는 tentative로 표시돼야 한다"
    assert "해당될 경우" in by_action[urgent].urgency_note
    own = next(a for a in by_action if "대출성 상품 청약철회" in a)
    assert not by_action[own].tentative, "1순위 카드는 확정으로 남아야 한다"


def test_alt_category_does_not_duplicate_shared_rules():
    """공통 룰(_common)은 2순위 때문에 두 번 뜨지 않는다."""
    items = compute_deadlines(
        "불완전판매", "2026년 7월 20일", today=TODAY, alt_category="투자상품",
    )
    actions = [i.action for i in items]
    assert len(actions) == len(set(actions)), f"중복 카드: {actions}"


def test_alt_category_same_as_category_is_noop():
    base = compute_deadlines("보험", "2026년 7월 20일", today=TODAY)
    same = compute_deadlines("보험", "2026년 7월 20일", today=TODAY, alt_category="보험")
    assert [i.action for i in base] == [i.action for i in same]
    assert not any(i.tentative for i in same)


def test_product_type_confirmed_card_is_not_marked_tentative(monkeypatch):
    """상품유형이 확정한 청약철회는 2순위와 겹쳐도 확정으로 남는다.

    제46조①이 기간을 정한 카드다. 2순위 유형에도 같은 카드가 있었다는 이유로
    「해당될 수 있음」을 달면 조문이 정한 기간을 추측처럼 보여주게 된다.
    """
    from app.agent import deadlines as dl
    from app.agent import product_type

    monkeypatch.setattr(product_type, "infer", lambda t, top=2: ["보장성 상품"])
    items = dl.compute_deadlines(
        "불완전판매", "2026년 7월 20일", today=TODAY,
        situation_text="보험 가입할 때 설명을 못 들었어요", alt_category="보험",
    )
    card = next(i for i in items if "보장성 상품 청약철회" in i.action)
    assert not card.tentative, "상품유형이 정한 카드가 추측으로 표시됐다"
    assert "해당될 경우" not in card.urgency_note


def test_law_citation_is_cached(monkeypatch):
    """같은 근거 질의로 검색을 두 번 돌리지 않는다.

    `basis_query`는 RULES 테이블의 고정 상수인데, 캐시가 없을 때는 기한 카드마다
    하이브리드 검색(임베딩 + FAISS + BM25)을 새로 돌렸다. 카드가 4~6장이라
    질의 한 건에 그만큼이 사용자 대기 시간으로 얹혔다 — 실측 0.99초 → 0.20초.

    이 캐시가 조용히 빠지면 지연이 5배로 돌아오는데 기능은 멀쩡해서 눈에 띄지
    않는다. 그래서 호출 횟수로 고정한다.
    """
    from app.agent import deadlines as dl

    calls: list[str] = []
    real = dl.retrieve

    def counting(kind, query, k=5):
        calls.append(query)
        return real(kind, query, k=k)

    dl._law_citation.cache_clear()
    monkeypatch.setattr(dl, "retrieve", counting)
    dl.compute_deadlines("보험", "2026년 7월 20일", today=TODAY)
    first = len(calls)
    assert first > 0, "근거 조문을 아예 찾지 않았다"

    dl.compute_deadlines("보험", "2026년 7월 20일", today=TODAY)
    assert len(calls) == first, f"같은 질의로 검색을 다시 돌렸다 ({first} → {len(calls)})"
    dl._law_citation.cache_clear()
