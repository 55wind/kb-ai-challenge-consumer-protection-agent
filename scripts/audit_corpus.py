"""코퍼스가 **정말로 온전한지** 전수 점검한다.

`validate_corpus.py`는 스키마만 본다 — 필드가 있고 출처가 비어 있지 않은지.
그것만으로는 실제 사고를 못 잡는다. 이 서비스에서 데이터가 깨지는 방식은 따로 있다.

  1. **인덱스와 코퍼스가 어긋난다.** 코퍼스를 다시 만들고 인덱스를 안 만들면
     검색이 옛 ID를 돌려주고, `_get_records[rid]`에서 KeyError가 난다.
     사용자에게는 500으로 보인다. 레코드 수가 같아도 ID 집합이 다르면 같은 일이
     생기므로 **집합으로** 비교한다.
  2. **그래프가 없는 문서를 가리킨다.** 결정례를 지우고 그래프를 안 만들면
     근거 확장이 유령 노드를 물어 온다.
  3. **정답지가 없는 문서를 가리킨다.** 골든셋·분류 정답지가 사라진 ID를 들고
     있으면 영원히 맞출 수 없는 문항이 되는데 점수만 조용히 떨어진다.
  4. **테스트 픽스처가 실코퍼스에 섞인다.** `[TEST-FIXTURE]` 출처가 운영
     데이터에 들어가면 심사자가 클릭했을 때 존재하지 않는 출처를 본다.
  5. **본문이 비었거나 잘렸다.** 파서가 조용히 실패하면 제목만 있고 본문이 없는
     레코드가 남는다. 검색에는 걸리는데 답변에 쓸 내용이 없다.

전부 통과하면 종료 코드 0. 하나라도 걸리면 1.

사용:
    python scripts/audit_corpus.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402
from app.corpus.loader import load_corpus  # noqa: E402

KINDS = ("laws", "cases", "interpretations", "procedures", "kb_products")
BAR = "─" * 68
# 본문이 이보다 짧으면 파서가 조용히 실패했을 가능성이 있다.
MIN_BODY = 40


def _body(r) -> str:
    for attr in ("text", "reasoning", "facts"):
        v = getattr(r, attr, None)
        if v:
            return v
    return ""


def main() -> int:
    problems: list[str] = []
    loaded: dict[str, list] = {}

    print(f"{BAR}\n① 코퍼스 — 스키마·ID·출처·본문")
    for kind in KINDS:
        try:
            records = load_corpus(kind)
        except Exception as e:  # noqa: BLE001
            problems.append(f"[{kind}] 로드 실패: {type(e).__name__}: {e}")
            print(f"  {kind:<17} 로드 실패 — {e}")
            continue
        loaded[kind] = records

        ids = [r.id for r in records]
        dup = [i for i, c in Counter(ids).items() if c > 1]
        fixtures = [r.id for r in records if "[TEST-FIXTURE]" in r.source]
        no_url = [r.id for r in records
                  if not str(r.source_url).startswith(("http://", "https://"))]
        thin = [r.id for r in records if len(_body(r)) < MIN_BODY]

        flags = []
        if dup:
            flags.append(f"ID 중복 {len(dup)}건")
            problems.append(f"[{kind}] ID 중복: {dup[:5]}")
        if fixtures:
            flags.append(f"테스트 픽스처 {len(fixtures)}건")
            problems.append(f"[{kind}] 실코퍼스에 픽스처: {fixtures[:5]}")
        if no_url:
            flags.append(f"출처 URL 이상 {len(no_url)}건")
            problems.append(f"[{kind}] source_url이 http로 시작하지 않음: {no_url[:5]}")
        if thin:
            flags.append(f"본문 빈약 {len(thin)}건")
            problems.append(f"[{kind}] 본문 {MIN_BODY}자 미만: {thin[:5]}")
        print(f"  {kind:<17} {len(records):>4}건   {'· '.join(flags) if flags else '이상 없음'}")

    print(f"\n{BAR}\n② 인덱스 — 코퍼스와 같은 문서를 담고 있는가")
    for kind, records in loaded.items():
        ids_path = settings.INDEX_DIR / f"{kind}.ids.json"
        if not ids_path.exists():
            problems.append(f"[{kind}] 인덱스 없음 — build_index.py 필요")
            print(f"  {kind:<17} 인덱스 없음")
            continue
        idx_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        corpus_ids = {r.id for r in records}
        missing = corpus_ids - set(idx_ids)     # 코퍼스에 있는데 색인 안 됨
        stale = set(idx_ids) - corpus_ids       # 색인에는 있는데 코퍼스에서 사라짐
        try:
            import faiss

            vecs = faiss.read_index(str(settings.INDEX_DIR / f"{kind}.faiss")).ntotal
        except Exception:  # noqa: BLE001
            vecs = -1
        note = []
        if missing:
            note.append(f"미색인 {len(missing)}건")
            problems.append(f"[{kind}] 색인 안 된 레코드: {sorted(missing)[:5]}")
        if stale:
            note.append(f"유령 {len(stale)}건")
            problems.append(f"[{kind}] 코퍼스에 없는 ID가 색인에 있음 "
                            f"(검색 시 KeyError): {sorted(stale)[:5]}")
        if vecs != len(idx_ids):
            note.append(f"벡터 {vecs} ≠ ID {len(idx_ids)}")
            problems.append(f"[{kind}] FAISS 벡터 수와 ID 수 불일치")
        print(f"  {kind:<17} 색인 {len(idx_ids):>4} · 벡터 {vecs:>4}   "
              f"{'· '.join(note) if note else '코퍼스와 일치'}")

    print(f"\n{BAR}\n③ 그래프 — 실재하는 문서만 가리키는가")
    gpath = settings.INDEX_DIR / "graph.json"
    if not gpath.exists():
        problems.append("그래프 없음 — build_graph.py 필요")
        print("  graph.json 없음")
    else:
        g = json.loads(gpath.read_text(encoding="utf-8"))
        known = {r.id for recs in loaded.values() for r in recs}
        # 그래프에는 코퍼스 레코드 id를 그대로 쓰지 **않는** 노드가 둘 있다.
        # 둘 다 설계이므로 유령으로 세면 안 된다.
        #
        #   `prod:<상품명>`  상품 노드는 상품 단위다. 코퍼스는 조항 청크 단위라
        #                   1:1로 대응하지 않는다(상품 3종 ↔ 청크 43개).
        #   `ref:<법령>`     코퍼스 밖 법령(민법·자본시장법 등)의 참조 노드.
        #                   `meta.stub=True`로 표시하고 국가법령정보센터 링크를 단다.
        typed = {"case": "cases", "law": "laws", "procedure": "procedures"}
        ghost = [n["id"] for n in g["nodes"]
                 if n["type"] in typed and n["id"] not in known
                 and not n["id"].startswith(("ref:", "prod:"))
                 and not n.get("meta", {}).get("stub")]
        node_ids = {n["id"] for n in g["nodes"]}
        dangling = [e["id"] for e in g["edges"]
                    if e["src"] not in node_ids or e["dst"] not in node_ids]
        no_prov = [e["id"] for e in g["edges"] if not e.get("provenance", {}).get("method")]
        if ghost:
            problems.append(f"[graph] 코퍼스에 없는 문서 노드 {len(ghost)}건: {ghost[:5]}")
        if dangling:
            problems.append(f"[graph] 끊어진 엣지 {len(dangling)}건: {dangling[:5]}")
        if no_prov:
            problems.append(f"[graph] provenance 없는 엣지 {len(no_prov)}건")
        print(f"  노드 {len(g['nodes'])} · 엣지 {len(g['edges'])}   "
              f"{'이상 없음' if not (ghost or dangling or no_prov) else '문제 있음'}")
        print(f"  provenance: {dict(Counter(e['provenance']['method'] for e in g['edges']))}")

    print(f"\n{BAR}\n④ 평가 데이터 — 실재하는 문서를 가리키는가")
    for name, path, kind_field, id_field in [
        ("골든셋", settings.DATA_DIR / "goldenset.jsonl", "kind", "expected_ids"),
    ]:
        if not path.exists():
            problems.append(f"{name} 파일 없음: {path}")
            continue
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        bad = []
        for e in rows:
            known = {r.id for r in loaded.get(e[kind_field], [])}
            bad += [i for i in e[id_field] if i not in known]
        if bad:
            problems.append(f"[{name}] 없는 문서를 가리킴: {bad[:5]}")
        print(f"  {name:<12} {len(rows):>4}문항   {'이상 없음' if not bad else f'없는 ID {len(bad)}건'}")

    for name, path in [("분류 정답지", settings.DATA_DIR / "classifier_goldset.jsonl"),
                       ("분류 프로브", settings.DATA_DIR / "classifier_probe.jsonl"),
                       ("사례집 색인", settings.DATA_DIR / "fss_case_index.jsonl")]:
        if not path.exists():
            problems.append(f"{name} 파일 없음: {path.name}")
            print(f"  {name:<12}    - 없음")
            continue
        n = len([x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()])
        print(f"  {name:<12} {n:>4}건   존재")

    print(f"\n{BAR}")
    if problems:
        print(f"문제 {len(problems)}건")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("전부 통과 — 코퍼스·인덱스·그래프·평가 데이터가 서로 일치합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
