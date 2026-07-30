import json

import httpx
import pytest

from app.config import settings
from app.llm.ollama_client import LLMUnavailable, OllamaClient


def _client_with(handler) -> OllamaClient:
    return OllamaClient(base_url="http://test", transport=httpx.MockTransport(handler))


def test_generate_returns_response():
    def handler(request):
        body = json.loads(request.content)
        # 모델명을 리터럴로 박지 않는다 — 재는 것은 "설정된 모델을 그대로 보내는가"다
        assert body["model"] == settings.GEN_MODEL
        assert body["stream"] is False
        return httpx.Response(200, json={"response": "답변입니다"})

    assert _client_with(handler).generate("sys", "질문") == "답변입니다"


def test_schema_is_sent_as_format_and_wins_over_json_mode():
    """스키마를 주면 format에 스키마가 실린다 — `"json"`은 스키마를 보장하지 않는다.

    `format: "json"`으로는 EXAONE 4.0 1.2B가 요구한 키를 무시하고
    `{"type": "response", "text": ...}`을 냈다(스키마 준수 0%). 스키마를 넘기면
    Ollama가 디코딩을 제약해 그 키만 나올 수 있다. 이 경로가 조용히 끊기면
    준수율이 0으로 돌아가므로 고정한다.
    """
    schema = {"type": "object", "properties": {"level": {"type": "string", "enum": ["red"]}}}
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"response": '{"level": "red"}'})

    _client_with(handler).generate("sys", "질문", json_mode=True, schema=schema)
    assert seen["format"] == schema, "format에 스키마가 실려야 한다"
    assert seen["format"] != "json", "json_mode보다 스키마가 우선해야 한다"


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


def test_num_gpu_absent_by_default():
    """미지정이면 옵션을 보내지 않는다 — Ollama가 하드웨어를 보고 정하게 둔다."""
    def handler(request):
        assert "num_gpu" not in json.loads(request.content)["options"]
        return httpx.Response(200, json={"response": "ok"})

    _client_with(handler).generate("sys", "질문")


@pytest.mark.parametrize("mode", ["generate", "stream"])
def test_num_gpu_zero_forces_cpu(monkeypatch, mode):
    """OLLAMA_NUM_GPU=0이면 두 경로 모두 CPU 전용으로 요청한다.

    generate에만 붙고 stream에 빠지면, CPU 전용 사이트 검증이 스트리밍 경로를
    통과시켜 버린다 — 실제 답변은 전부 stream으로 나간다.
    """
    from app.llm import ollama_client

    monkeypatch.setattr(ollama_client.settings, "OLLAMA_NUM_GPU", 0)

    seen = {}

    def handler(request):
        seen.update(json.loads(request.content)["options"])
        return httpx.Response(200, json={"response": "ok", "done": True})

    c = _client_with(handler)
    if mode == "generate":
        c.generate("sys", "질문")
    else:
        list(c.stream("sys", "질문"))
    assert seen["num_gpu"] == 0
