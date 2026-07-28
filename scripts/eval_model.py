"""EXAONE 3.5 2.4B가 이 시스템에서 맡은 일을 얼마나 해내는지 측정한다.

측정 대상을 "모델이 똑똑한가"로 잡으면 안 된다. 이 시스템은 흐름 제어·계산·검증을
전부 코드가 하고 모델에게는 좁은 작업만 맡긴다. 그러므로 재야 할 것은
**맡긴 작업의 수행률**이다.

  1. 유형 분류 정확도  — 금감원이 라벨링한 결정례 98건이 정답지
  2. JSON 형식 준수율  — 깨지면 폴백이 돌아 분류 품질이 떨어진다
  3. 분류 일관성       — 같은 입력에 같은 답을 내는가 (temperature 0.2)
  4. 인용 준수율       — 근거 [n]을 붙이는가. 이 시스템의 핵심 보장
  5. 작업별 속도       — 어느 작업이 사용자 대기의 원인인가

정답지 주의: 결정례 facts는 법률 문어체다. 실제 소비자가 쓰는 구어체 민원과
분포가 다르므로, 분류 정확도는 대리 지표로 읽어야 한다. 이 한계는 결과에도 적는다.
"""
import argparse
import collections
import io
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BAR = "─" * 62


def _cases(limit: int) -> list[tuple[str, str]]:
    """(민원 텍스트, 정답 유형). facts 앞부분을 민원 진술로 쓴다."""
    rows = [json.loads(x) for x in (ROOT / "data/corpus/cases.jsonl").open(encoding="utf-8")]
    out = []
    for r in rows:
        facts = (r.get("facts") or "").strip()
        cat = r.get("category")
        if len(facts) < 60 or not cat:
            continue
        out.append((facts[:600], cat))
    random.Random(20260728).shuffle(out)  # 유형 편중을 흩되 재현 가능하게
    return out[:limit]


def _goldenset() -> list[dict]:
    return [json.loads(x) for x in (ROOT / "data/goldenset.jsonl").open(encoding="utf-8")]


def _deadline_set(category: str) -> frozenset[str]:
    """그 유형에서 안내되는 기한 항목. 유형이 틀리면 이것이 달라진다."""
    from app.agent.deadlines import RULES
    return frozenset(r.action for r in RULES.get(category, []) + RULES["_common"])


def eval_classification(llm, limit: int) -> dict:
    from app.agent import classifier

    data = _cases(limit)
    hit = fallback = 0
    lat: list[float] = []
    confusion: collections.Counter = collections.Counter()
    # 오분류의 결과를 나눈다. 유형은 라우팅과 기한 룰을 고르므로, 같은 오분류라도
    # 기한 안내가 그대로면 사용자에게 보이는 피해가 없다.
    harmless = missing = extra = misrouted = 0

    for text, gold in data:
        t = time.perf_counter()
        c = classifier.classify(text, llm)
        lat.append(time.perf_counter() - t)
        if c.confidence == "low":
            fallback += 1
        if c.category == gold:
            hit += 1
            continue

        confusion[(gold, c.category)] += 1
        want, got = _deadline_set(gold), _deadline_set(c.category)
        if c.category == "절차문의" or gold == "절차문의":
            misrouted += 1          # 상담이 아니라 절차 안내 경로로 빠진다
        elif want == got:
            harmless += 1           # 기한 안내가 같다 — 화면 결과가 사실상 동일
        elif want - got:
            missing += 1            # 안내했어야 할 기한이 빠진다 ← 가장 위험
        else:
            extra += 1              # 무관한 기한이 덧붙는다

    n = len(data)
    return {
        "n": n,
        "정확도": hit / n,
        "JSON 준수율": 1 - fallback / n,
        "지연_중앙값": statistics.median(lat),
        "오분류": confusion.most_common(4),
        "무해": harmless, "기한누락": missing, "기한과다": extra, "경로이탈": misrouted,
    }


def eval_consistency(llm, rounds: int) -> dict:
    """같은 입력을 반복했을 때 같은 답이 나오는 비율."""
    from app.agent import classifier

    probes = [t for t, _ in _cases(6)]
    stable = 0
    for text in probes:
        answers = [classifier.classify(text, llm).category for _ in range(rounds)]
        if len(set(answers)) == 1:
            stable += 1
    return {"n": len(probes), "회차": rounds, "일관성": stable / len(probes)}


def eval_citation(llm) -> dict:
    """근거 [n]을 붙이는 비율. 1차 통과 / 재시도 후 통과 / 최종 실패."""
    from app.agent import answerer
    from app.agent.answerer import UngroundedAnswer
    from app.search.retriever import NoEvidenceFound, retrieve

    first = retried = failed = skipped = 0
    lat: list[float] = []
    tok: list[int] = []

    for row in _goldenset():
        try:
            hits = retrieve(row["kind"], row["query"], k=4)
        except NoEvidenceFound:
            skipped += 1
            continue

        t = time.perf_counter()
        try:
            ans = answerer.answer(row["query"], hits, llm)   # 재시도 없는 1차 시도
            first += 1
            tok.append(len(ans.text))
            lat.append(time.perf_counter() - t)
        except UngroundedAnswer:
            try:
                ans = answerer.answer_with_retry(row["query"], hits, llm)
                retried += 1
                tok.append(len(ans.text))
                lat.append(time.perf_counter() - t)
            except UngroundedAnswer:
                failed += 1

    n = first + retried + failed
    return {
        "n": n, "검색실패_제외": skipped,
        "1차 통과": first, "재시도 후 통과": retried, "최종 실패": failed,
        "1차 준수율": first / n if n else 0,
        "최종 준수율": (first + retried) / n if n else 0,
        "지연_중앙값": statistics.median(lat) if lat else 0,
        "답변_평균글자": statistics.mean(tok) if tok else 0,
    }


def eval_speed(llm) -> dict:
    """작업별 지연. 사용자가 기다리는 시간이 어디서 오는지 본다."""
    from app.agent import answerer, classifier
    from app.agent import slots as slots_mod
    from app.search.retriever import NoEvidenceFound, retrieve

    rows = _goldenset()[:5]
    marks: dict[str, list[float]] = collections.defaultdict(list)

    for row in rows:
        t = time.perf_counter()
        classifier.classify(row["query"], llm)
        marks["분류"].append(time.perf_counter() - t)

        t = time.perf_counter()
        slots_mod.extract_slots([{"role": "user", "content": row["query"]}], llm)
        marks["슬롯 추출"].append(time.perf_counter() - t)

        try:
            hits = retrieve(row["kind"], row["query"], k=4)
        except NoEvidenceFound:
            continue
        t = time.perf_counter()
        try:
            answerer.answer(row["query"], hits, llm)
        except Exception:
            pass
        marks["답변 생성"].append(time.perf_counter() - t)

    return {k: statistics.median(v) for k, v in marks.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40, help="분류 평가 표본 수")
    ap.add_argument("--rounds", type=int, default=3, help="일관성 반복 횟수")
    ap.add_argument("--quick", action="store_true", help="표본을 줄여 빠르게")
    a = ap.parse_args()
    if a.quick:
        a.cases, a.rounds = 12, 2

    from app.llm.ollama_client import OllamaClient
    llm = OllamaClient()

    print(f"EXAONE 3.5 2.4B 품질·속도 측정  (모델: {llm.model})")
    print(BAR)

    print(f"\n[1] 유형 분류 — 금감원 라벨 결정례 {a.cases}건")
    c = eval_classification(llm, a.cases)
    print(f"  정확도        {c['정확도']:.1%}  ({int(c['정확도'] * c['n'])}/{c['n']})")
    print(f"  JSON 준수율   {c['JSON 준수율']:.1%}  (미달 시 임베딩 폴백)")
    print(f"  지연 중앙값   {c['지연_중앙값']:.2f}초")
    if c["오분류"]:
        print("  주요 오분류:")
        for (gold, got), cnt in c["오분류"]:
            print(f"    {gold} → {got}  {cnt}건")
    wrong = c["n"] - int(c["정확도"] * c["n"])
    print(f"  오분류 {wrong}건이 사용자 화면에 미치는 영향:")
    print(f"    무해(기한 동일)     {c['무해']}건")
    print(f"    기한 누락           {c['기한누락']}건  ← 안내했어야 할 기한이 빠진다")
    print(f"    무관한 기한 추가    {c['기한과다']}건")
    print(f"    절차 경로로 이탈    {c['경로이탈']}건")

    print(f"\n[2] 분류 일관성 — 같은 입력 {a.rounds}회 반복")
    k = eval_consistency(llm, a.rounds)
    print(f"  동일 응답률   {k['일관성']:.1%}  ({k['n']}개 질의)")

    print("\n[3] 인용 준수 — 골든셋 전 문항에 답변 생성")
    z = eval_citation(llm)
    print(f"  1차 준수율    {z['1차 준수율']:.1%}  ({z['1차 통과']}/{z['n']})")
    print(f"  최종 준수율   {z['최종 준수율']:.1%}  (재시도 {z['재시도 후 통과']}건 포함)")
    print(f"  최종 실패     {z['최종 실패']}건  ← 답변 대신 '근거 없음' 안내")
    print(f"  지연 중앙값   {z['지연_중앙값']:.2f}초 · 평균 {z['답변_평균글자']:.0f}자")

    print("\n[4] 작업별 속도")
    for name, sec in eval_speed(llm).items():
        print(f"  {name:<12}{sec:>7.2f}초")

    print(f"\n{BAR}")
    print("정답지 한계: 결정례 facts는 법률 문어체다. 실제 소비자 구어체와 분포가")
    print("달라 분류 정확도는 대리 지표다. 인용 준수율은 실제 파이프라인 그대로다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
