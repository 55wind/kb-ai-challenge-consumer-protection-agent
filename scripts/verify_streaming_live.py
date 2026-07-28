"""실서버에서 답변 토큰이 실제로 흐르는지 확인.

측정하려는 것: 첫 글자가 화면에 닿기까지 얼마나 걸리는가.
스트리밍 이전에는 답변이 완성될 때까지(수십 초) 아무것도 안 보였다.
"""
import io
import json
import sys
import time

import httpx

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8021"
QUERY = sys.argv[2] if len(sys.argv) > 2 else "펀드 가입할 때 설명의무는 어떻게 되나요?"


def main() -> int:
    c = httpx.Client(timeout=600)
    sid = c.post(f"{BASE}/api/session").json()["session_id"]
    t0 = time.time()
    first_delta = None
    deltas = 0
    answer_at = None
    discarded = False

    with c.stream("POST", f"{BASE}/api/chat",
                  json={"session_id": sid, "text": QUERY}) as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event:
                if event == "answer_delta":
                    deltas += 1
                    if first_delta is None:
                        first_delta = time.time() - t0
                        tok = json.loads(line.split(":", 1)[1]).get("text", "")
                        print(f"  첫 조각 도착: {first_delta:.1f}초  \"{tok}\"")
                elif event == "answer_discard":
                    discarded = True
                    print("  잠정 표시 폐기 (인용 없어 재시도)")
                elif event == "answer":
                    answer_at = time.time() - t0
                elif event == "done":
                    break

    print(f"\n질문: {QUERY}")
    print(f"  흘린 조각: {deltas}개")
    if first_delta and answer_at:
        print(f"  첫 글자까지: {first_delta:.1f}초")
        print(f"  답변 완성까지: {answer_at:.1f}초")
        print(f"  → 스트리밍이 없었다면 {answer_at:.1f}초 동안 빈 화면이었다")
    elif answer_at:
        print(f"  조각 없이 {answer_at:.1f}초 만에 완성 (스트리밍 미동작)")
    else:
        print("  답변이 오지 않았다")
    if discarded:
        print("  (재시도 발생 — 잠정 표시는 화면에서 지워졌다)")
    return 0 if deltas else 1


if __name__ == "__main__":
    raise SystemExit(main())
