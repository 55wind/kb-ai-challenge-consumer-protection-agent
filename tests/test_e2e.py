"""E2E: 실제 코퍼스·인덱스로 서버 전체 흐름 검증 (LLM만 스텁).

임베딩 모델 로드 때문에 느리다 → @pytest.mark.slow.
사전 조건: data/corpus/*.jsonl + data/indexes/* 존재 (scripts/build_index.py).
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.server import db
from app.server.api import AppContext, create_app

pytestmark = pytest.mark.slow


class ScriptedLLM:
    """호출 목적을 프롬프트 내용으로 식별해 그럴듯한 고정 응답 반환."""

    model = "scripted"

    def generate(self, system, prompt, json_mode=False, timeout=120, schema=None):
        if "분류기" in system:
            return json.dumps({"category": "불완전판매", "is_dispute": False}, ensure_ascii=False)
        if "사실관계 정리" in system:
            return json.dumps({"who": "본인", "when": "2025년", "product": "펀드",
                               "incident": "설명 없이 가입", "demand": "배상"}, ensure_ascii=False)
        if "위반 심사" in system:
            return json.dumps({"level": "red", "reason": "원금 보장 표현"}, ensure_ascii=False)
        if "사례 분석" in system:
            return "유사 사례들을 보면 설명의무 위반 여부가 쟁점이었습니다."
        return "근거 문서에 따르면 설명의무가 있습니다[1]."


TEST_TOKEN = "e2e-admin-token"


@pytest.fixture(scope="module")
def client():
    import os

    from app.server.security import reset_admin_token_for_tests

    missing = [k for k in ("laws", "cases", "procedures")
               if not (settings.INDEX_DIR / f"{k}.faiss").exists()]
    if missing:
        pytest.skip(f"인덱스 없음: {missing} — scripts/build_index.py 먼저 실행")
    os.environ["ADMIN_TOKEN"] = TEST_TOKEN
    reset_admin_token_for_tests()
    db_path = settings.DATA_DIR / "e2e_test.db"
    conn = db.connect(db_path)
    db.sync_documents(conn)
    ctx = AppContext(llm=ScriptedLLM(), conn=conn)
    yield TestClient(create_app(ctx))
    conn.close()
    # WAL 모드라 -wal/-shm이 함께 남고, 윈도우에서는 닫은 직후 잠금이 풀리기 전
    # unlink가 PermissionError를 낸다. 정리 실패로 스위트를 빨갛게 만들지 않는다.
    for suffix in ("", "-wal", "-shm"):
        try:
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
        except OSError:
            pass
    reset_admin_token_for_tests()
    os.environ.pop("ADMIN_TOKEN", None)


def _sid(c) -> str:
    return c.post("/api/session").json()["session_id"]


def _events(resp) -> dict[str, list]:
    out: dict[str, list] = {}
    event = None
    for line in resp.text.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event:
            out.setdefault(event, []).append(json.loads(line.split(":", 1)[1]))
    return out


def test_chat_general_flow_with_real_retrieval(client):
    resp = client.post("/api/chat", json={
        "session_id": _sid(client),
        "text": "펀드 가입할 때 설명의무는 어떻게 되나요?",
    })
    ev = _events(resp)
    assert "disclaimer" in ev and "answer" in ev and "citations" in ev
    cited = ev["citations"][0]["items"]
    assert any("금융소비자" in c["source"] for c in cited)  # 실제 법령 출처


def test_scan_flow_real_law_citation(client):
    resp = client.post("/api/chat", json={
        "session_id": _sid(client),
        "text": "이 상품은 원금 100% 보장에 확정 수익을 드립니다.",
        "mode": "scan",
    })
    ev = _events(resp)
    report = ev["scan_report"][0]
    assert report["input_kind"] == "document"
    red = [i for i in report["items"] if i["level"] == "red"]
    assert red and red[0]["law_citation"]["title"].startswith("제22조")


def test_sms_batch_scan_e2e(client):
    resp = client.post("/api/chat", json={
        "session_id": _sid(client),
        "text": "[Web발신] 금융감독원입니다. 안전계좌로 이체하세요\n\n"
                "(광고) 커피 쿠폰 도착 무료수신거부\n\n"
                "대출 승인! 보증금 먼저 입금하세요 010-1111-2222",
        "mode": "scan",
    })
    ev = _events(resp)
    report = ev["scan_report"][0]
    assert report["input_kind"] == "sms_batch"
    assert report["summary"]["total"] == 3
    assert report["summary"]["red"] + report["summary"]["yellow"] >= 2
    assert any("118" in a for a in report["summary"]["actions"])


def test_transcript_to_draft_e2e(client):
    sid = _sid(client)
    resp = client.post("/api/chat", json={
        "session_id": sid,
        "text": "[00:10] 직원: 이 펀드는 원금 보장이라 안전합니다\n"
                "[00:20] 고객: 그래요?\n"
                "[00:30] 직원: 무조건 수익 나니까 오늘 가입하세요",
        "mode": "scan",
    })
    ev = _events(resp)
    tr = ev["transcript_report"][0]
    assert tr["summary"]["total"] == 2  # 직원 발화만
    assert len(tr["items"]) >= 1

    # 같은 세션에서 분쟁 슬롯을 채우고 초안 생성 → 별첨 포함
    ctx_state = client.app.state.ctx.state(sid)
    ctx_state.slots_filled = {"who": "본인", "when": "2026년 7월", "product": "펀드",
                              "incident": "원금보장 설명만 듣고 가입", "demand": "손해배상"}
    ctx_state.category = "불완전판매"
    draft = client.post(f"/api/draft/{sid}", json={"respondent": "○○은행"}).json()
    assert "별첨" in draft["markdown"]
    assert "원금 보장이라 안전합니다" in draft["markdown"]


def test_phishing_dispute_emits_urgent_deadline(client):
    class PhishingLLM(ScriptedLLM):
        def generate(self, system, prompt, json_mode=False, timeout=120, schema=None):
            if "분류기" in system:
                return json.dumps({"category": "보이스피싱·전자금융사기", "is_dispute": True},
                                  ensure_ascii=False)
            if "사실관계 정리" in system:
                return json.dumps({"who": "본인", "when": "오늘", "product": "예금계좌",
                                   "incident": "보이스피싱으로 이체", "demand": "환급"},
                                  ensure_ascii=False)
            return super().generate(system, prompt, json_mode, timeout)

    ctx = client.app.state.ctx
    original = ctx.llm
    ctx.llm = PhishingLLM()
    try:
        resp = client.post("/api/chat", json={
            "session_id": _sid(client),
            "text": "방금 보이스피싱을 당해서 돈이 이체됐어요",
        })
    finally:
        ctx.llm = original
    ev = _events(resp)
    items = ev["deadlines"][0]["items"]
    assert items[0]["urgency"] == "urgent"
    assert "지급정지" in items[0]["action"]


def test_admin_endpoints_after_traffic(client):
    headers = {"X-Admin-Token": TEST_TOKEN}
    tickets = client.get("/api/admin/tickets", headers=headers).json()["tickets"]
    assert any(t["category"] == "불완전판매" for t in tickets)
    docs = client.get("/api/admin/documents", params={"kind": "cases"},
                      headers=headers).json()["documents"]
    assert len(docs) >= 30  # 실제 결정례 최소 요건


def test_admin_documents_include_kb_products(client):
    """KB 상품 필터가 비어 있으면 KB 특화 근거가 화면에서 사라진다.

    검색 인덱스에는 있는데 documents 테이블에만 빠져 있던 회귀를 고정한다.
    """
    headers = {"X-Admin-Token": TEST_TOKEN}
    docs = client.get("/api/admin/documents", params={"kind": "kb_products"},
                      headers=headers).json()["documents"]
    assert docs, "관리자 지식베이스에 KB 상품 문서가 하나도 없다"
    assert all(d["kind"] == "kb_products" for d in docs)
    # 제목이 원시 ID로 새지 않고 상품명으로 읽혀야 한다
    assert any("KB" in d["title"] or "예금" in d["title"] or "적금" in d["title"]
               for d in docs), f"제목이 상품명으로 표시되지 않는다: {docs[0]['title']}"
    # 실데이터 원칙 — 전부 출처를 가진다
    assert all(d["source"] and d["source_url"] for d in docs)
