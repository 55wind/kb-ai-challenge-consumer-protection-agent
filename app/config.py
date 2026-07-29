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
    # 하이브리드 검색 결합 상수. 근거는 scripts/tune_retrieval.py 실측(골든셋 81문항).
    #
    # RRF_K: 원 논문의 60은 수백~수천 건짜리 검색 결과를 합칠 때의 값이다. 여기서는
    #   후보가 15개뿐이라 60을 쓰면 1위(1/61)와 5위(1/65)의 점수 차가 6%밖에 안 난다.
    #   순위가 사실상 무의미해져서, 두 목록에 겹쳐 나오기만 하면 이긴다 — 정답 조문이
    #   벡터 3위인데도 등록요건 조문에 밀리는 일이 실제로 일어났다.
    # BM25_WEIGHT: 소비자는 "상담해준 사람이 은행 직원인지 대리점인지 밝히지 않았어요"
    #   라고 쓰지 "고지의무"라고 쓰지 않는다. 겹치는 낱말이 없으니 BM25는 "금융상품·
    #   판매·대리·중개"가 잔뜩 들어간 **등록요건 조문**을 집어 온다. 실측에서도
    #   BM25 단독 93.8% < 벡터 단독 97.5%였다. 그래도 0으로 두지 않는 이유는
    #   섞었을 때가 벡터 단독보다 나았기 때문이다 — 조문 번호나 고유명사가 든
    #   질문은 BM25가 잡아낸다.
    #
    #   0.25가 BM25를 무력화하지는 않는다는 점은 분명히 해 둔다. BM25 1위라는
    #   최대 가산(0.25/11 = 0.023)은 벡터 5위(0.067)를 벡터 2위 위로 올릴 만큼은
    #   된다. 노리는 것은 "BM25를 죽이는 것"이 아니라 **끌어올리는 폭을 몇 계단
    #   안으로 묶는 것**이다. 벡터가 제대로 찾은 근거가 상위 5건 밖으로 밀려나지만
    #   않으면 된다 — 답변은 상위 4~5건을 근거로 쓰기 때문이다.
    RRF_K: int = _env_int("RRF_K", 10)
    BM25_WEIGHT: float = _env_float("BM25_WEIGHT", 0.25)
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
