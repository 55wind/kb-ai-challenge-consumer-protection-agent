import json

import pytest

from app.corpus.loader import load_corpus
from app.corpus.schema import CaseRecord, CorpusValidationError, InterpretationDoc


def _case_kwargs(**overrides):
    base = dict(
        id="c1",
        title="[TEST-FIXTURE] 사례",
        facts="사실관계",
        issues="쟁점",
        reasoning="판단",
        decision="인용",
        decision_date="2023-01-01",
        category="불완전판매",
        source="[TEST-FIXTURE] pytest",
        source_url="about:test",
    )
    base.update(overrides)
    return base


def test_case_requires_source():
    with pytest.raises(CorpusValidationError):
        CaseRecord.validated(**_case_kwargs(source="", source_url=""))


def test_case_valid():
    r = CaseRecord.validated(**_case_kwargs(decision="기각"))
    assert r.decision == "기각"


def test_case_rejects_unknown_decision():
    with pytest.raises(Exception):
        CaseRecord.validated(**_case_kwargs(decision="보류"))


def test_load_corpus_roundtrip(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_case_kwargs()) + "\n", encoding="utf-8")
    records = load_corpus("cases", corpus_dir=tmp_path)
    assert len(records) == 1
    assert records[0].id == "c1"


def test_load_corpus_fails_on_missing_source(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_case_kwargs(source="")) + "\n", encoding="utf-8")
    with pytest.raises(CorpusValidationError):
        load_corpus("cases", corpus_dir=tmp_path)


def _interp_kwargs(**over):
    base = dict(
        id="i1", source="[TEST-FIXTURE] pytest", source_url="about:test",
        doc_type="FAQ", topic="Ⅰ. 시험", title="시험 문항인가?",
        text="□ 그렇다.", cited_articles=["제19조"],
    )
    return {**base, **over}


def test_interpretation_requires_source():
    with pytest.raises(CorpusValidationError):
        InterpretationDoc.validated(**_interp_kwargs(source="", source_url=""))


def test_interpretation_rejects_unknown_doc_type():
    """FAQ와 해설만 허용한다 — 화면이 이 값으로 출처 표기를 달리하기 때문이다.

    조문(법)과 유권해석(당국 견해)은 효력이 다르다. 새 값이 조용히 들어오면
    구분이 무너진 채로 인용 화면에 나간다.
    """
    with pytest.raises(Exception):
        InterpretationDoc.validated(**_interp_kwargs(doc_type="유권해석"))


def test_real_interpretations_corpus_loads():
    """실제 코퍼스가 스키마를 통과하는지. 파싱 스크립트 회귀를 여기서 잡는다."""
    records = load_corpus("interpretations")
    assert len(records) > 100, "유권해석 코퍼스가 비었거나 파싱이 깨졌다"
    # 허용 값을 여기에 또 적지 않는다 — 스키마가 하나뿐인 정의여야 한다.
    # 예전에는 {"FAQ", "해설"}을 박아 두어, 스키마가 "안내"를 받도록 늘어난 뒤에도
    # 테스트만 실패했다. 재야 할 것은 "코퍼스가 스키마를 벗어나지 않는가"다.
    allowed = set(InterpretationDoc.model_fields["doc_type"].annotation.__args__)
    assert {r.doc_type for r in records} <= allowed
    assert {r.doc_type for r in records}, "doc_type이 비었다"
    assert all(r.title.strip() and r.text.strip() for r in records)


def test_case_category_records_who_decided_it():
    """유형을 누가 정했는지가 레코드에 남아야 한다.

    결정례의 민원 유형은 **원문에 없다** — 금감원은 결정서에 유형을 붙여
    배포하지 않는다. 그런데 우리 키워드 규칙이 만든 값을 문서가 "기관이 분류한
    실데이터"라고 설명해 왔고, 금감원 사례집과 대조 가능한 17건에서 3건(18%)이
    틀렸다. 추정을 기관 분류인 척하지 않으려면 이 구분이 레코드에 있어야 한다.
    """
    records = load_corpus("cases")
    assert len(records) >= 100
    sources = {r.category_source for r in records}
    assert sources <= {"금융감독원 사례집", "키워드 규칙(추정)"}, sources
    # 사례집과 이어 붙인 건이 실제로 있어야 한다 — 0이면 색인이 깨진 것이다.
    assert sum(r.category_source == "금융감독원 사례집" for r in records) >= 10


def test_fss_case_index_is_present_and_labeled_by_the_regulator():
    """금감원 사례집 색인이 있어야 분류 정답지를 기관 라벨로 유지할 수 있다."""
    import json
    from pathlib import Path

    from app.config import settings

    path = settings.DATA_DIR / "fss_case_index.jsonl"
    assert path.exists(), "build_fss_case_index.py를 실행하세요"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) >= 300, f"사례집 색인이 {len(rows)}건뿐이다 — 목차 파싱이 깨졌다"
    assert all(r["source"].startswith("금융감독원") for r in rows)
    assert len({r["chapter"] for r in rows}) >= 15, "금감원 분류 체계가 덜 잡혔다"


def test_no_duplicate_ids_in_any_corpus():
    """어느 코퍼스에도 id가 겹치면 안 된다.

    `retriever._get_records`가 `{r.id: r}` dict라 **겹치면 하나가 다른 하나를
    덮어쓴다.** 인덱스에는 두 벡터가 다 남으므로 검색은 되는데, 조회에서 엉뚱한
    레코드가 나온다.

    실제로 났던 일: KB 예금거래기본약관의 `제9조`와 `제9조의2`가 같은 id를
    받았다(라벨 정규식이 가지번호 `의2`를 버렸다). 그 결과 제9조(이자)는
    검색에서 사라졌고, 제9조 벡터에 맞은 질의에 **제9조의2(휴면예금) 본문이
    인용문으로** 나갔다. 근거를 보여준다는 이 시스템의 약속이 깨지는 자리다.
    """
    from collections import Counter

    for kind in ("laws", "cases", "interpretations", "procedures", "kb_products"):
        ids = [r.id for r in load_corpus(kind)]
        dup = [i for i, c in Counter(ids).items() if c > 1]
        assert not dup, f"{kind}에 중복 id: {dup}"


def test_index_ids_match_corpus():
    """인덱스가 코퍼스와 같은 문서를 담고 있어야 한다.

    코퍼스만 다시 만들고 인덱스를 안 만들면 검색이 사라진 id를 돌려주고
    `_get_records[rid]`에서 KeyError가 난다 — 사용자에게는 500이다.
    개수가 같아도 집합이 다르면 같은 일이 나므로 집합으로 본다.
    """
    import json

    from app.config import settings

    for kind in ("laws", "cases", "interpretations", "procedures", "kb_products"):
        path = settings.INDEX_DIR / f"{kind}.ids.json"
        if not path.exists():
            continue  # 인덱스 미빌드 환경(CI 등)에서는 건너뛴다
        indexed = set(json.loads(path.read_text(encoding="utf-8")))
        corpus = {r.id for r in load_corpus(kind)}
        assert indexed == corpus, (
            f"{kind}: 색인에만 {sorted(indexed - corpus)[:3]} · "
            f"코퍼스에만 {sorted(corpus - indexed)[:3]}"
        )


def test_product_goldset_labels_come_from_the_statute():
    """상품유형 정답지의 라벨은 **조문 적용**이어야 한다.

    이 프로젝트는 자체 라벨로 자체 판정을 채점하다 두 번 뒤집혔다(분류 정답지,
    프로브). 상품유형 정답지에서 같은 일이 나지 않게, 레코드마다 어느 조문으로
    정했는지가 남아 있는지 검사한다.
    """
    import json

    from app.config import settings

    path = settings.DATA_DIR / "product_goldset.jsonl"
    if not path.exists():
        import pytest as _pytest
        _pytest.skip("scripts/build_product_goldset.py 미실행")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) >= 25, f"{len(rows)}문항뿐이다 — 후보 수집이 깨졌다"
    assert {r["product_type"] for r in rows} <= {
        "예금성 상품", "대출성 상품", "투자성 상품", "보장성 상품"}, "제3조 4종을 벗어났다"
    for r in rows:
        assert r["label_basis"], f"라벨 근거가 없다: {r['query'][:40]}"
        # 근거 문자열에 조문 번호가 있어야 한다 — "우리가 보기에"가 아니라 조문이다
        assert any("제" in b and "조" in b for b in r["label_basis"]), r["label_basis"]
        assert len(r["label_basis"]) == 1, (
            f"낱말이 둘 이상 걸린 문항은 제3조 단서(중복 귀속)라 버려야 한다: {r['query'][:40]}")
