"""설정. 모든 값은 환경변수로 재정의 가능 — 은행 서버 배포 시 코드 수정 없이
모델·경로·용량을 바꾼다 (예: GEN_MODEL=kb-internal-llm OLLAMA_BASE_URL=http://gpu-node:11434).
"""
import os
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Settings(BaseModel):
    OLLAMA_BASE_URL: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    GEN_MODEL: str = _env_str("GEN_MODEL", "exaone3.5:2.4b")
    # GPU에 올릴 레이어 수. 미지정이면 Ollama가 하드웨어를 보고 정한다(권장).
    # 0 = CPU 전용 — GPU 없는 사이트 배포 전 검증, CPU 성능 실측에 쓴다.
    OLLAMA_NUM_GPU: int | None = (
        int(_env_str("OLLAMA_NUM_GPU", "")) if _env_str("OLLAMA_NUM_GPU", "") else None
    )
    EMBED_MODEL: str = _env_str("EMBED_MODEL", "nlpai-lab/KURE-v1")
    DATA_DIR: Path = Path(_env_str("DATA_DIR", str(ROOT / "data")))
    CORPUS_DIR: Path = Path(_env_str("DATA_DIR", str(ROOT / "data"))) / "corpus"
    INDEX_DIR: Path = Path(_env_str("DATA_DIR", str(ROOT / "data"))) / "indexes"
    DB_PATH: Path = Path(_env_str("DATA_DIR", str(ROOT / "data"))) / "app.db"
    WEB_DIST: Path = Path(_env_str("WEB_DIST", str(ROOT / "web" / "dist")))
    RELEVANCE_THRESHOLD: float = _env_float("RELEVANCE_THRESHOLD", 0.35)
    # 답변 토큰 스트리밍. CPU 추론 대기(수십 초)를 글자로 채운다.
    # false면 완성 후 한 번에 보내는 이전 동작 — 시연에서 불안하면 끄면 된다.
    ANSWER_STREAMING: bool = _env_str("ANSWER_STREAMING", "true").lower() != "false"
    # 컨텍스트 그래프 — 근거 확장(cites 엣지)과 큐레이션 반영
    # GRAPH_EXPANSION=false로 끄면 그래프 이전 동작으로 즉시 되돌아간다(킬 스위치).
    GRAPH_EXPANSION: bool = _env_str("GRAPH_EXPANSION", "true").lower() != "false"
    GRAPH_EXPANSION_MAX: int = _env_int("GRAPH_EXPANSION_MAX", 2)
    # 관리 인스턴스의 큐레이션이 고객 인스턴스에 반영되기까지의 지연(초)
    GRAPH_CURATION_TTL: float = _env_float("GRAPH_CURATION_TTL", 30.0)
    # 동시접속·보안 (하드닝 계획 Task 7·8)
    LLM_MAX_CONCURRENT: int = _env_int("LLM_MAX_CONCURRENT", 2)
    LLM_ACQUIRE_TIMEOUT: float = _env_float("LLM_ACQUIRE_TIMEOUT", 30.0)
    SESSION_TTL_SECONDS: int = _env_int("SESSION_TTL_SECONDS", 7200)
    SESSION_MAX: int = _env_int("SESSION_MAX", 1000)
    INPUT_MAX_CHARS: int = _env_int("INPUT_MAX_CHARS", 20000)
    RATE_LIMIT_PER_MINUTE: int = _env_int("RATE_LIMIT_PER_MINUTE", 30)
    CORS_ORIGINS: str = _env_str("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    # 리버스 프록시(TLS 종단) 뒤 배포 시 신뢰할 프록시 IP·CIDR 목록.
    # 등록된 프록시를 거친 요청에 한해 X-Forwarded-For를 클라이언트 IP로 채택한다.
    # 빈 값(기본)이면 헤더를 무시 — 위조로 레이트리밋·IP 허용목록을 우회할 수 없다.
    TRUSTED_PROXY_IPS: str = _env_str("TRUSTED_PROXY_IPS", "")
    # 관리 기능 분리 (정보보호): deny by default.
    # - 고객용 인스턴스: 기본값 그대로(false) → 관리 라우트 자체가 등록되지 않음
    # - 관리용 인스턴스: ADMIN_ENABLED=true + 내부망 전용 + ADMIN_IP_ALLOWLIST + ADMIN_TOKENS
    ADMIN_ENABLED: bool = _env_str("ADMIN_ENABLED", "false").lower() == "true"
    ADMIN_IP_ALLOWLIST: str = _env_str("ADMIN_IP_ALLOWLIST", "")  # 빈 값 = 위치 제한 없음(기동 시 경고)
    ADMIN_TOKENS: str = _env_str("ADMIN_TOKENS", "")  # "이름:시크릿,이름2:시크릿2"


settings = Settings()
