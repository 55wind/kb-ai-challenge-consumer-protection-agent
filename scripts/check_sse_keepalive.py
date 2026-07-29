"""SSE 스트림에 바이트가 언제 흐르는지 본다 (ping 포함).

이벤트 사이가 오래 비면 프록시·방화벽·백신이 연결을 끊고, 브라우저는 그것을
"network error"로 표시한다. 서버 쪽 로그에는 아무 오류가 없기 때문에 이 구간을
직접 봐야 원인을 알 수 있다.
"""
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
    t0 = time.time()
    last = t0
    gaps = []

    with c.stream("POST", f"{BASE}/api/chat",
                  json={"session_id": sid, "text": QUERY}) as r:
        for raw in r.iter_raw():
            if not raw:
                continue
            now = time.time()
            gap = now - last
            gaps.append(gap)
            head = raw[:48].decode("utf-8", "replace").replace("\r", "").replace("\n", "⏎")
            print(f"  [{now - t0:6.1f}s] +{gap:5.1f}s  {len(raw):5d}B  {head}")
            last = now

    print(f"\n총 {time.time() - t0:.1f}초 · 청크 {len(gaps)}개")
    worst = max(gaps) if gaps else 0
    print(f"가장 긴 무음 구간: {worst:.1f}초")
    if worst > 20:
        print("  ⚠ 20초 넘게 아무 바이트도 안 나간다. 중간 장비가 끊을 수 있는 구간이다.")
    else:
        print("  keep-alive가 흐르고 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
