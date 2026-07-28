"""보안 유틸: 서버 발급 세션, 관리자 토큰, 레이트리밋, PII 마스킹, 보안 헤더.

전부 표준 라이브러리 — 신규 의존성 없음 (하드닝 계획 Task 8).
"""
import os
import re
import secrets
import threading
import time

from fastapi import Header, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


def new_session_id() -> str:
    return secrets.token_urlsafe(24)


# ── PII 마스킹 (로그 저장용 — 세션 내 원문은 유지) ──────────────────
_RRN = re.compile(r"\d{6}-\d{7}")                       # 주민등록번호
_PHONE = re.compile(r"(01\d)-(\d{3,4})-(\d{4})")        # 휴대전화
_ACCOUNT = re.compile(r"\d{10,16}")                     # 계좌 추정(연속 숫자)
# 하이픈 구분 계좌(예: 123-456-789012). 날짜(2026-07-27, 8자리)를 삼키지 않도록
# 전체 자릿수가 계좌 범위(10~16)일 때만 마스킹한다.
_ACCOUNT_HYPHEN = re.compile(r"\d{2,6}(?:-\d{2,6}){1,3}")


def _mask_hyphenated_account(m: re.Match) -> str:
    s = m.group(0)
    digits = sum(c.isdigit() for c in s)
    if not 10 <= digits <= 16:
        return s
    out, seen = [], 0
    for c in s:
        if c.isdigit():
            seen += 1
            out.append(c if seen <= 4 else "*")
        else:
            out.append(c)
    return "".join(out)


def mask_pii(text: str) -> str:
    text = _RRN.sub("******-*******", text)
    text = _PHONE.sub(lambda m: f"{m.group(1)}-{'*' * len(m.group(2))}-{m.group(3)}", text)
    text = _ACCOUNT_HYPHEN.sub(_mask_hyphenated_account, text)
    text = _ACCOUNT.sub(lambda m: m.group(0)[:4] + "*" * (len(m.group(0)) - 4), text)
    return text


# ── 클라이언트 IP 해석 (리버스 프록시 뒤 배포 대응) ──────────────────
def _ip_in_any(host: str, networks: list[str]) -> bool:
    import ipaddress

    for entry in networks:
        if entry == host:
            return True
        try:
            if ipaddress.ip_address(host) in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def client_ip(request) -> str:
    """실제 클라이언트 IP.

    운영 가이드가 요구하는 리버스 프록시(TLS 종단) 뒤 배포에서 request.client.host는
    프록시 IP 하나로 고정되어 레이트리밋·IP 허용목록이 무력화된다. 그래서
    TRUSTED_PROXY_IPS에 등록된 프록시를 거쳐 온 요청에 한해 X-Forwarded-For를 신뢰한다.
    등록이 없으면(기본값) 헤더는 위조 가능하므로 무시한다 — fail-closed.
    """
    from app.config import settings

    peer = request.client.host if request.client else ""
    trusted = [e.strip() for e in settings.TRUSTED_PROXY_IPS.split(",") if e.strip()]
    if not trusted or not _ip_in_any(peer, trusted):
        return peer
    xff = request.headers.get("x-forwarded-for", "")
    # 오른쪽부터 훑어 신뢰 프록시 체인을 건너뛴 첫 홉이 실제 클라이언트다.
    for candidate in reversed([p.strip() for p in xff.split(",") if p.strip()]):
        if not _ip_in_any(candidate, trusted):
            return candidate
    return peer


# ── 레이트리밋 (인메모리 슬라이딩 윈도우) ────────────────────────────
class RateLimiter:
    def __init__(self, limit: int = 30, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self.window]
            if len(hits) >= self.limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True


# ── 관리자 토큰 (토큰별 신원 — 감사 추적용) ──────────────────────────
_admin_tokens: dict[str, str] | None = None  # secret → actor


def get_admin_tokens() -> dict[str, str]:
    """ADMIN_TOKENS("이름:시크릿,...") 우선, 없으면 ADMIN_TOKEN(신원 'admin'),
    둘 다 없으면 기동 시 생성해 콘솔에 1회 출력."""
    global _admin_tokens
    if _admin_tokens is None:
        tokens: dict[str, str] = {}
        multi = os.environ.get("ADMIN_TOKENS", "")
        for pair in multi.split(","):
            pair = pair.strip()
            if ":" in pair:
                actor, secret = pair.split(":", 1)
                if actor.strip() and secret.strip():
                    tokens[secret.strip()] = actor.strip()
        single = os.environ.get("ADMIN_TOKEN")
        if single:
            tokens.setdefault(single, "admin")
        if not tokens:
            generated = secrets.token_urlsafe(16)
            tokens[generated] = "admin"
            print(f"[보안] 관리자 토큰(자동 생성): {generated}")
        _admin_tokens = tokens
    return _admin_tokens


def get_admin_token() -> str:
    """(하위 호환) 첫 번째 관리자 시크릿."""
    return next(iter(get_admin_tokens()))


def reset_admin_token_for_tests() -> None:
    global _admin_tokens
    _admin_tokens = None


def resolve_admin_actor(token: str) -> str | None:
    """토큰 → 행위자 이름. 불일치 시 None (타이밍 안전 비교)."""
    matched = None
    for secret, actor in get_admin_tokens().items():
        if secrets.compare_digest(token, secret):
            matched = actor
    return matched


def check_admin_ip(client_host: str | None) -> bool:
    """ADMIN_IP_ALLOWLIST 검사. 빈 목록 = 허용(기동 시 경고). 항목은 IP·CIDR·리터럴 매치."""
    from app.config import settings

    allowlist = [e.strip() for e in settings.ADMIN_IP_ALLOWLIST.split(",") if e.strip()]
    if not allowlist:
        return True
    return _ip_in_any(client_host or "", allowlist)


def admin_guard(x_admin_token: str = Header(default="")) -> None:
    """(하위 호환) 신원 확인 없는 단순 가드 — api.py의 admin_access가 표준."""
    if resolve_admin_actor(x_admin_token) is None:
        raise HTTPException(401, "관리자 토큰이 올바르지 않습니다")


# ── 보안 헤더 ───────────────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
