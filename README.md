# 금융 소비자 보호 에이전트

제8회 KB AI Challenge 출품작 (현직자 Pick #4).
로컬 sLLM이 금융 민원을 이해하고, **실제 법령·금감원 분쟁조정 결정례에 근거해**
해결 방안과 절차를 안내하며, 불공정 상황에서 소비자의 권익 보호 행동을 돕는다.

> **권익침해를 탐지하고(레드플래그 스캐너), 승산을 보여주고(유사 실사례 전망), 행동으로 옮겨준다(신청서 초안 자동 생성).**

## 핵심 특징

- **100% 실데이터**: 금감원 분쟁조정 결정례 98건, 금소법·시행령·감독규정 217개 조문, 금융위·금감원 금소법 FAQ·설명자료 130건, FINE 금융용어사전 534개 — 가상·합성 사례를 일절 사용하지 않으며, 스키마 차원에서 출처(source/source_url)를 강제
- **완전 로컬 추론**: Ollama + EXAONE 3.5 2.4B (CPU, 4bit) — 민원 속 개인정보가 외부로 나가지 않음
- **하이브리드 오케스트레이션**: 흐름 제어는 결정론적 코드, sLLM은 이해·생성만 담당 (소형 모델의 약점을 구조로 보완)
- **근거 인용 강제**: 인용 없는 답변은 거부·재시도, 관련도 미달 시 추측 대신 "근거 없음" + 금감원 1332 안내
- **책임 있는 전망**: 유사 사례의 인용/기각 분포를 사실 서술만 — 확률 수치 생성을 코드로 금지

## 실행 (운영 모드 — 은행이 사용자에게 제공하는 단일 서비스)

```bash
# 0. 사전 준비 (최초 1회)
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
cd web && npm install && npm run build && cd ..
python scripts/build_index.py && python scripts/build_graph.py

# 1. 기동 (Ollama·모델·빌드 자동 확인)
.\run.ps1          # Windows  (Linux/macOS: ./run.sh)
```

- 고객(금융소비자): `http://<서버>:8000/chat` — 상담, 검진 3종, 골든타임, 신청서 초안
- 담당자(KB 소비자보호부): `http://<서버>:8000/admin` — 티켓·지식베이스·지식 그래프·로그
- 운영: 운영 가이드 문서 (환경변수·모니터링·데이터 갱신·백업·장애 대응) — 아래 "문서" 참조
- 은행 서버 배포: `docker compose --profile init run --rm ollama-init && docker compose --profile init run --rm app-init && docker compose up -d`
- 개발 모드: `uvicorn ... --port 8000` + `cd web && npm run dev` (5173). 관리 화면(`/admin`)은 기본 비활성(deny by default)이라 `ADMIN_ENABLED=true`를 함께 지정한다

## 검증

```bash
.venv/Scripts/python -m pytest tests/ -q        # 207 tests
.venv/Scripts/python scripts/eval_goldenset.py  # 검색 품질: 골든셋 81문항 hit@5 100%
.venv/Scripts/python scripts/validate_corpus.py # 실데이터 출처 검증
.venv/Scripts/python scripts/tune_retrieval.py  # 검색기 비중·RRF 상수 재측정
.venv/Scripts/python scripts/compare_models.py  # 생성 모델 후보 비교 (라이선스 대응)
```

골든셋 hit@5 100%는 **회귀 감시선이지 일반화 성능이 아니다.** 81문항은 코퍼스에서
뽑아 쓴 질문이고 RRF 상수 2개를 이 문항들로 골랐다. 코퍼스·모델을 바꿨을 때
검색이 나빠졌는지 알아채는 용도로 쓴다.

## 구조

```
app/
  agent/        분류기·슬롯필링·답변생성·사례전망·레드플래그·신청서초안·오케스트레이터
  corpus/       스키마(출처 강제)·로더·HWP 파서
  search/       KURE-v1 임베딩·FAISS+BM25 인덱스·RRF 검색
  llm/          Ollama 클라이언트
  server/       FastAPI(SSE)·SQLite
data/corpus/    구조화 실데이터 (laws/cases/interpretations/procedures/kb_products/glossary)
scripts/        수집·구축·인덱싱·평가
web/            React 프론트 (/chat, /admin)
```

## 문서

기획서·기술명세서·사용법·아키텍처·운영 가이드는 **이 저장소에 포함하지 않는다.**
용량이 큰 PDF·PPTX이고 코드와 갱신 주기가 다르다. 별도 산출물 패키지
(`제출_산출물_v3.zip`)로 관리하며, 실행에 필요한 내용은 `공유_안내.md`에 정리돼 있다.

## 라이선스 유의

현재 생성 모델 **EXAONE 3.5는 비상업(NC) 라이선스**다. 대회 참가는 라이선스에
명시적으로 허용돼 있으나(제2.1(a)조), **상업 서비스 배포는 금지**된다(제3.1조).
EXAONE 4.0도 같다. 실서비스로 전환하려면 상업 이용이 가능한 모델
(Mi:dm 2.0 Mini — MIT, Kanana 1.5 — Apache 2.0)로 바꿔야 하며, `GEN_MODEL`
환경변수 한 줄로 교체된다. 임베딩(KURE-v1)·런타임(Ollama)·인덱스(FAISS)는 모두
MIT라 제약이 없다. 상세는 산출물 패키지의 `10_라이선스_검토`.

데이터 출처: 금융감독원(분쟁조정결정례·절차 안내·금소법 설명자료·FINE 용어사전),
금융위원회(금소법 FAQ 1~3차·10問10答), 국가법령정보센터(금소법 3법령 외).
본 서비스의 답변은 법률 자문이 아닌 정보 제공입니다.
