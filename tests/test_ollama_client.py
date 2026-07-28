import json

import httpx
import pytest

from app.llm.ollama_client import LLMUnavailable, OllamaClient


def _client_with(handler) -> OllamaClient:
    return OllamaClient(base_url="http://test", transport=httpx.MockTransport(handler))


def test_generate_returns_response():
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "exaone3.5:2.4b"
        assert body["stream"] is False
        return httpx.Response(200, json={"response": "답변입니다"})

    assert _client_with(handler).generate("sys", "질문") == "답변입니다"


def test_generate_json_mode_sets_format():
    def handler(request):
        body = json.loads(request.content)
        assert body["format"] == "json"
        return httpx.Response(200, json={"response": "{\"a\": 1}"})

    assert _client_with(handler).generate("sys", "질문", json_mode=True) == '{"a": 1}'


def test_generate_raises_llm_unavailable_on_connect_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailable, match="Ollama 실행 여부"):
        _client_with(handler).generate("sys", "질문")


# ── 타임아웃 (H-6) ──────────────────────────────────────────────────
# CPU 추론에서 가장 흔한 실패. 변환되지 않으면 스트림이 무음으로 끊긴다.
def test_generate_read_timeout_becomes_llm_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("read timed out")

    with pytest.raises(LLMUnavailable, match="제한 시간"):
        _client_with(handler).generate("sys", "질문")


def test_stream_read_timeout_becomes_llm_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("read timed out")

    with pytest.raises(LLMUnavailable, match="제한 시간"):
        list(_client_with(handler).stream("sys", "질문"))


def test_generate_http_error_becomes_llm_unavailable():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMUnavailable):
        _client_with(handler).generate("sys", "질문")
