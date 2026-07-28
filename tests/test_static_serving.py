"""프로덕션 정적 서빙: 단일 오리진에서 SPA 라우팅 폴백 검증."""
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.server import db
from app.server.api import AppContext, create_app


class StubLLM:
    model = "stub"

    def generate(self, *a, **k):
        return ""


@pytest.fixture
def client(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>KB-APP</body></html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setattr(settings, "WEB_DIST", dist)
    ctx = AppContext(llm=StubLLM(), conn=db.connect(tmp_path / "s.db"))
    return TestClient(create_app(ctx))


def test_spa_routes_serve_index(client):
    for path in ("/", "/chat", "/admin"):
        r = client.get(path)
        assert r.status_code == 200
        assert "KB-APP" in r.text


def test_static_asset_served(client):
    assert client.get("/assets/app.js").status_code == 200


def test_api_prefix_not_swallowed(client):
    assert client.get("/api/nonexistent").status_code == 404
    assert client.get("/api/health").status_code == 200


# ── 경로 탈출 회귀 (C-1) ────────────────────────────────────────────
# dist 밖 파일(민원 DB·소스·설정)이 인증 없이 새면 정보보호 서사 전체가 무너진다.
# 어떤 인코딩으로도 dist 밖을 읽을 수 없어야 하며, 실패는 index.html 폴백이어야 한다.
TRAVERSAL_VECTORS = [
    "../secret.txt",
    "..%2fsecret.txt",
    "..%2F..%2Fsecret.txt",
    "%2e%2e/secret.txt",
    "%2e%2e%2fsecret.txt",
    "%252e%252e%252fsecret.txt",       # 이중 인코딩
    "..\\secret.txt",
    "..%5csecret.txt",                 # 백슬래시(윈도우)
    "dist/../../secret.txt",
    "assets/../../secret.txt",
]


@pytest.mark.parametrize("vector", TRAVERSAL_VECTORS)
def test_path_traversal_blocked(client, tmp_path, vector):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-DB", encoding="utf-8")
    r = client.get(f"/{vector}")
    assert r.status_code == 200          # SPA 폴백
    assert "TOP-SECRET-DB" not in r.text  # 내용은 절대 새지 않는다
    assert "KB-APP" in r.text


def test_absolute_path_blocked(client, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-DB", encoding="utf-8")
    r = client.get(f"/{secret.as_posix()}")
    assert "TOP-SECRET-DB" not in r.text
