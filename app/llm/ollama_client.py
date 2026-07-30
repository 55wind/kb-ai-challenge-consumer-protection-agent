"""Ollama HTTP 클라이언트. 외부 LLM API 호출 금지 — 로컬 추론 전용."""
import json
from typing import Iterator

import httpx

from app.config import settings


class LLMUnavailable(Exception):
    pass


_TIMEOUT_MESSAGE = (
    "답변 생성이 제한 시간을 넘겨 중단했습니다(CPU 추론 지연). "
    "잠시 후 다시 시도하시거나, 문의 내용을 조금 짧게 나눠 보내주세요."
)


def _options(**extra) -> dict:
    """Ollama 추론 옵션. NUM_GPU를 지정하면 GPU에 올릴 레이어 수를 강제한다.

    기본은 미지정 — Ollama가 하드웨어를 보고 알아서 정한다. 0을 주면 CPU 전용이
    되는데, GPU 없는 사이트에 배포하기 전 같은 장비에서 CPU 동작을 검증하거나
    CPU 성능을 실측할 때 쓴다.
    """
    opts = {"temperature": 0.2, "num_ctx": 4096, **extra}
    if settings.OLLAMA_NUM_GPU is not None:
        opts["num_gpu"] = settings.OLLAMA_NUM_GPU
    return opts


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.GEN_MODEL
        self._client = httpx.Client(base_url=self.base_url, transport=transport, timeout=None)

    def generate(self, system: str, prompt: str, json_mode: bool = False,
                 timeout: float = 120, schema: dict | None = None) -> str:
        """schema를 주면 Ollama가 그 JSON Schema로 **디코딩을 제약**한다.

        `format: "json"`은 "문법이 맞는 JSON"만 보장한다. 스키마는 보장하지 않아
        모델이 키 이름을 제 마음대로 정할 수 있다. 실제로 EXAONE 4.0 1.2B는
        `{"category": ..., "is_dispute": ...}`를 요구했더니 `{"type": "response",
        "text": "..."}`를 냈다 — 문법은 멀쩡하지만 앱 입장에서는 파싱 실패다.

        스키마를 넘기면 Ollama가 문법을 제약해 그 키·타입·enum만 나올 수 있게
        만든다. 같은 질의에서 준수 실패가 사라지고, 군말을 생성할 수 없으므로
        오히려 빨라진다(3.4초 → 0.7초 실측).

        json_mode는 스키마를 모르는 호출(자유형 JSON)을 위해 남긴다.
        """
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": _options(),
        }
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"
        try:
            resp = self._client.post("/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
        # TimeoutException이 먼저 — ReadTimeout(CPU 추론 지연에서 가장 흔한 실패)이
        # RequestError로 흡수되어 '연결 실패'로 오인되지 않게 한다.
        except httpx.TimeoutException as e:
            raise LLMUnavailable(_TIMEOUT_MESSAGE) from e
        except httpx.RequestError as e:
            raise LLMUnavailable(
                f"Ollama에 연결할 수 없습니다({self.base_url}). Ollama 실행 여부 확인: 'ollama serve'"
            ) from e
        except httpx.HTTPStatusError as e:
            raise LLMUnavailable(f"Ollama 오류: {e.response.status_code} {e.response.text[:200]}") from e
        return resp.json().get("response", "")

    def stream(self, system: str, prompt: str, timeout: float = 300) -> Iterator[str]:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": True,
            "options": _options(),
        }
        try:
            with self._client.stream("POST", "/api/generate", json=payload, timeout=timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if token := chunk.get("response"):
                        yield token
                    if chunk.get("done"):
                        break
        except httpx.TimeoutException as e:
            raise LLMUnavailable(_TIMEOUT_MESSAGE) from e
        except httpx.RequestError as e:
            raise LLMUnavailable(
                f"Ollama에 연결할 수 없습니다({self.base_url}). Ollama 실행 여부 확인: 'ollama serve'"
            ) from e
        except httpx.HTTPStatusError as e:
            raise LLMUnavailable(
                f"Ollama 오류: {e.response.status_code}"
            ) from e
