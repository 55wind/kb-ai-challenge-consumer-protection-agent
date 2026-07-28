# 금융 소비자 보호 에이전트 — 앱 이미지 (은행 온프레미스 배포용)
# 프론트는 빌드 산출물(web/dist)을 포함해 단일 컨테이너로 서빙한다.
FROM node:20-slim AS web
WORKDIR /build
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /srv
ENV PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=0
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ app/
COPY scripts/ scripts/
COPY data/corpus/ data/corpus/
COPY data/goldenset.jsonl data/goldenset.jsonl
COPY --from=web /build/dist web/dist
# 인덱스·그래프는 볼륨 또는 초기화 단계에서 생성:
#   docker compose run app python scripts/build_index.py && python scripts/build_graph.py
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import httpx; r=httpx.get('http://localhost:8000/api/health',timeout=4); exit(0 if r.json().get('ok') else 1)"
CMD ["python", "-m", "uvicorn", "app.server.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
