# 금융 소비자 보호 에이전트 — 운영 기동 스크립트 (Windows)
# 사용: .\run.ps1  (옵션: $env:ADMIN_TOKEN, $env:PORT 사전 설정)
$ErrorActionPreference = "Stop"
$PORT = if ($env:PORT) { $env:PORT } else { "8000" }

Write-Host "[1/4] Ollama 확인..."
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/version" -UseBasicParsing -TimeoutSec 2 | Out-Null
} catch {
    Write-Host "  Ollama 기동 중..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
}

Write-Host "[2/4] 모델 확인 (exaone3.5:2.4b)..."
$models = ollama list
if ($models -notmatch "exaone3.5:2.4b") {
    Write-Host "  모델 다운로드 중 (~1.6GB)..."
    ollama pull exaone3.5:2.4b
}

Write-Host "[3/4] 프론트 빌드 확인..."
if (-not (Test-Path "web\dist\index.html")) {
    Write-Host "  web/dist 없음 — 빌드 실행..."
    Push-Location web
    npm run build
    Pop-Location
}

Write-Host "[4/4] 서버 기동 (포트 $PORT)..."
if (-not $env:ADMIN_TOKEN) {
    Write-Host "  ADMIN_TOKEN 미설정 — 기동 로그에 자동 생성 토큰이 1회 출력됩니다." -ForegroundColor Yellow
}
# 단일 인스턴스 시연 모드: 관리 화면을 함께 띄운다.
# (운영 배포는 고객용/관리용 인스턴스를 분리하고 관리용에만 ADMIN_ENABLED=true를 준다)
if (-not $env:ADMIN_ENABLED) { $env:ADMIN_ENABLED = "true" }
Write-Host ""
Write-Host "  고객 화면:   http://localhost:$PORT/chat" -ForegroundColor Green
Write-Host "  담당자 화면: http://localhost:$PORT/admin" -ForegroundColor Green
Write-Host "  헬스체크:    http://localhost:$PORT/api/health" -ForegroundColor Green
Write-Host ""
.venv\Scripts\python -m uvicorn app.server.api:create_app --factory --host 0.0.0.0 --port $PORT
