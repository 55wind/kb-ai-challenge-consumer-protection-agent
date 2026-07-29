"""특정 질문에서 나는 오류를 재현한다. 이벤트를 전부 그대로 찍는다."""
import io
import sys
import time

import httpx

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8031"
QUERY = sys.argv[2] if len(sys.argv) > 2 else "불공정 계약을 당했을 때는 어떻게 해야돼?"


def main() -> int:
    c = httpx.Client(timeout=600)
    sid = c.post(f"{BASE}/api/session").json()["session_id"]
    print(f"질문: {QUERY}\n")
    t0 = time.time()
    ev = None
    deltas = 0
    try:
        with c.stream("POST", f"{BASE}/api/chat",
                      json={"session_id": sid, "text": QUERY}) as r:
            print(f"HTTP {r.status_code}")
            for line in r.iter_lines():
                if line.startswith("event:"):
                    ev = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and ev:
                    body = line.split(":", 1)[1].strip()
                    if ev == "answer_delta":
                        deltas += 1
                        continue
                    print(f"  [{time.time() - t0:6.1f}s] {ev}: {body[:220]}")
    except Exception as e:
        print(f"\n예외: {type(e).__name__}: {e}")
        print(f"  끊긴 시각 {time.time() - t0:.1f}초 · 마지막 이벤트 {ev} · 조각 {deltas}개")
        return 1
    print(f"\n완료 {time.time() - t0:.1f}초 · 조각 {deltas}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
