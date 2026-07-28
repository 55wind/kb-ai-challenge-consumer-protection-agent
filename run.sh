#!/usr/bin/env bash
# 금융 소비자 보호 에이전트 — 운영 기동 스크립트 (Linux/macOS)
set -euo pipefail
PORT="${PORT:-8000}"

echo "[1/4] Ollama 확인..."
if ! curl -s -m 2 http://localhost:11434/api/version >/dev/null; then
  echo "  Ollama 기동 중..."
  nohup ollama serve >/dev/null 2>&1 &
  sleep 4
fi

echo "[2/4] 모델 확인 (exaone3.5:2.4b)..."
ollama list | grep -q "exaone3.5:2.4b" || ollama pull exaone3.5:2.4b

echo "[3/4] 프론트 빌드 확인..."
if [ ! -f web/dist/index.html ]; then
  (cd web && npm run build)
fi

echo "[4/4] 서버 기동 (포트 $PORT)..."
[ -z "${ADMIN_TOKEN:-}" ] && echo "  ADMIN_TOKEN 미설정 — 기동 로그의 자동 생성 토큰을 확인하세요."
# 단일 인스턴스 시연 모드: 관리 화면을 함께 띄운다.
# (운영 배포는 고객용/관리용 인스턴스를 분리하고 관리용에만 ADMIN_ENABLED=true를 준다)
export ADMIN_ENABLED="${ADMIN_ENABLED:-true}"
echo "  고객:   http://localhost:$PORT/chat"
echo "  담당자: http://localhost:$PORT/admin"
exec python -m uvicorn app.server.api:create_app --factory --host 0.0.0.0 --port "$PORT"
