"""분류 정확도를 올릴 후보 두 가지를 실제로 만들어 재 본다.

`diagnose_classifier.py`가 오차를 이렇게 갈랐다 (결정례 60건).

    오답 11건 = 후보 생성 실패 6건 + sLLM 택일 실패 5건

두 실패는 처방이 다르다. 여기서 각각에 하나씩 대 본다.

**후보 생성 (B).** 지금 유형 벡터는 사람이 쓴 한 줄 설명을 임베딩한 것이다
("불완전판매: 펀드, 신탁, 투자상품 등을 가입할 때 위험 설명을…"). 그런데 들어오는
질의는 민원 진술문이다. 설명문과 진술문은 문체가 달라 벡터 공간에서 멀다.
**실제 라벨이 붙은 결정례의 중심(centroid)을 유형 벡터로 쓰면** 질의와 같은
분포에 놓인다. 라벨은 금감원이 붙인 실데이터라 창작이 아니다.

  누수 주의: 평가 대상 건으로 만든 중심으로 그 건을 맞추면 부정행위다. 그래서
  **leave-one-out** — i번 건을 평가할 때 그 유형의 중심에서 i번을 뺀다.

**택일 (C).** 지금 프롬프트는 후보를 **이름만** 준다(`['불완전판매', '보험']`).
모델은 그 이름이 무슨 뜻인지 스스로 추측해야 한다. `CATEGORIES`에 이미 설명이
있으니 함께 주면 될 일이다.

사용:
    python scripts/experiment_classifier.py --cases 60           # 후보 회수율만 (빠름)
    python scripts/experiment_classifier.py --cases 60 --with-llm  # 최종 정확도까지
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.agent.classifier import CATEGORIES  # noqa: E402
from app.llm.ollama_client import OllamaClient  # noqa: E402
from app.search import embedder  # noqa: E402
from scripts.eval_model import _cases  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BAR = "─" * 66
NAMES = list(CATEGORIES)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def _all_labeled_cases() -> list[tuple[str, str]]:
    """라벨이 붙은 결정례 전부. 중심 계산용이라 평가 표본과 별개로 읽는다."""
    out = []
    for line in (ROOT / "data/corpus/cases.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        facts = (r.get("facts") or "").strip()
        if len(facts) >= 60 and r.get("category"):
            out.append((facts[:600], r["category"]))
    return out


def build_vectors():
    """설명 벡터와, 유형별 결정례 합·개수를 돌려준다 (leave-one-out에 필요)."""
    desc = _unit(embedder.embed([f"{n}: {d}" for n, d in CATEGORIES.items()]))

    labeled = _all_labeled_cases()
    vecs = _unit(embedder.embed([t for t, _ in labeled]))
    sums = {n: np.zeros(vecs.shape[1], dtype=np.float32) for n in NAMES}
    counts = {n: 0 for n in NAMES}
    by_text = {}
    for (text, cat), v in zip(labeled, vecs):
        if cat in sums:
            sums[cat] += v
            counts[cat] += 1
        by_text[text] = (cat, v)
    return desc, sums, counts, by_text


def category_matrix(desc, sums, counts, exclude=None, blend=0.5):
    """유형 벡터 행렬. 결정례가 있는 유형은 중심과 설명을 섞고, 없으면 설명만 쓴다.

    exclude=(유형, 벡터)를 주면 그 건을 중심에서 뺀다 (leave-one-out).
    """
    rows = []
    for i, name in enumerate(NAMES):
        s, c = sums[name].copy(), counts[name]
        if exclude and exclude[0] == name:
            s, c = s - exclude[1], c - 1
        if c <= 0:
            rows.append(desc[i])          # 결정례가 없는 유형 — 설명만
            continue
        centroid = _unit(s / c)
        rows.append(_unit(blend * centroid + (1 - blend) * desc[i]))
    return np.vstack(rows)


def recall_at(mat_fn, data, ks=(1, 2, 3)) -> dict:
    hits = {k: 0 for k in ks}
    for text, gold in data:
        qv = _unit(embedder.embed([text])[0])
        order = [NAMES[i] for i in np.argsort(mat_fn(text, gold) @ qv)[::-1]]
        r = order.index(gold) + 1
        for k in ks:
            hits[k] += r <= k
    return {k: hits[k] / len(data) for k in ks}


def classify_with(text, cands, llm, with_desc: bool) -> str:
    system = (
        "너는 금융 민원 분류기다. 반드시 JSON으로만 답한다. "
        '형식: {"category": "<후보 중 하나>", "is_dispute": true|false}. '
        "is_dispute는 금융회사와의 다툼·피해구제 요구(분쟁성)이면 true, 단순 정보 문의면 false."
    )
    if with_desc:
        listed = "\n".join(f"- {c}: {CATEGORIES[c]}" for c in cands)
        prompt = f"민원 내용: {text}\n\n후보 유형:\n{listed}\n\n어느 유형인가?"
    else:
        prompt = f"민원 내용: {text}\n\n후보 유형: {cands}\n어느 유형인가?"
    try:
        got = json.loads(llm.generate(system, prompt, json_mode=True, timeout=60)).get("category", "")
        return got if got in CATEGORIES else cands[0]
    except Exception:  # noqa: BLE001
        return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--blend", type=float, default=0.5,
                    help="중심 대 설명 비율 (1.0이면 중심만)")
    args = ap.parse_args()

    data = _cases(args.cases)
    print(f"표본 {len(data)}건 · 중심 혼합비 {args.blend}\n")
    desc, sums, counts, by_text = build_vectors()
    print("유형별 결정례 수:", {n: c for n, c in counts.items() if c})

    base_mat = desc

    def loo_mat(text, gold):
        ex = by_text.get(text)
        return category_matrix(desc, sums, counts,
                               exclude=(ex[0], ex[1]) if ex else None, blend=args.blend)

    print(f"\n{BAR}\n① 후보 회수율")
    a = recall_at(lambda t, g: base_mat, data)
    b = recall_at(loo_mat, data)
    print(f"{'':<28}{'recall@1':<12}{'recall@2':<12}{'recall@3':<12}")
    print(f"{'A 설명만 (현재)':<28}" + "".join(f"{a[k]:.1%}".ljust(12) for k in (1, 2, 3)))
    print(f"{'B 결정례 중심 혼합':<28}" + "".join(f"{b[k]:.1%}".ljust(12) for k in (1, 2, 3)))

    if not args.with_llm:
        return 0

    print(f"\n{BAR}\n② 최종 정확도 (후보 2개)")
    llm = OllamaClient()
    variants = [
        ("A 현재 (설명만 · 이름만 제시)", lambda t, g: base_mat, False),
        ("C 후보 설명을 프롬프트에 포함", lambda t, g: base_mat, True),
        ("B 결정례 중심 혼합", loo_mat, False),
        ("D  B + C", loo_mat, True),
    ]
    for label, mat_fn, with_desc in variants:
        hit = 0
        for text, gold in data:
            qv = _unit(embedder.embed([text])[0])
            order = [NAMES[i] for i in np.argsort(mat_fn(text, gold) @ qv)[::-1]]
            hit += classify_with(text, order[:2], llm, with_desc) == gold
        print(f"  {label:<34}{hit}/{len(data)}  ({hit / len(data):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
