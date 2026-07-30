from app.agent.slots import SLOT_SCHEMA, extract_slots


class StubLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, system, prompt, json_mode=False, timeout=120, schema=None):
        return self.response


def test_extract_all_slots():
    llm = StubLLM(
        '{"who": "본인", "when": "2025년 3월", "product": "사모펀드",'
        ' "incident": "원금 손실 위험 설명 없이 가입", "demand": "손해배상"}'
    )
    slots = extract_slots([{"role": "user", "content": "..."}], llm)
    assert slots.missing == []
    assert slots.next_question is None
    assert set(slots.filled) == set(SLOT_SCHEMA)


def test_extract_partial_slots_asks_next():
    llm = StubLLM('{"who": "본인", "when": "", "product": "펀드", "incident": "", "demand": ""}')
    slots = extract_slots([{"role": "user", "content": "..."}], llm)
    assert "when" in slots.missing
    assert slots.next_question is not None
    assert "언제" in slots.next_question  # 첫 누락 슬롯(when) 질문


def test_extract_handles_bad_json():
    slots = extract_slots([{"role": "user", "content": "..."}], StubLLM("not json"))
    assert slots.missing == SLOT_SCHEMA
    assert slots.next_question is not None
