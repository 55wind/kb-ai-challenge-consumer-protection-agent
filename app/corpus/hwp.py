"""HWP 5.0 (OLE 컨테이너) 본문 텍스트 추출.

BodyText/Section* 스트림을 zlib 해제 후 레코드 단위로 순회하며
HWPTAG_PARA_TEXT(tag 67)의 UTF-16LE 텍스트를 모은다.
인라인/확장 컨트롤 문자(코드 1~23 일부)는 8워드를 차지하므로 건너뛴다.
"""
import struct
import zlib
from pathlib import Path

import olefile

HWPTAG_PARA_TEXT = 67
# 확장·인라인 컨트롤: 코드 뒤에 7워드(14바이트)의 부가 데이터가 붙는다
_CTRL_16BYTE = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


class HwpParseError(Exception):
    pass


def _para_text(chunk: bytes) -> str:
    out = []
    i = 0
    n = len(chunk) - 1
    while i < n:
        (code,) = struct.unpack_from("<H", chunk, i)
        if code in (10, 13):
            out.append("\n")
            i += 2
        elif code >= 32:
            out.append(chr(code))
            i += 2
        elif code in _CTRL_16BYTE:
            i += 16
        else:
            i += 2
    return "".join(out)


def extract_hwp_text(path: Path) -> str:
    try:
        ole = olefile.OleFileIO(str(path))
    except OSError as e:
        raise HwpParseError(f"{path.name}: OLE 컨테이너가 아님 ({e})") from e
    try:
        if not ole.exists("FileHeader"):
            raise HwpParseError(f"{path.name}: HWP FileHeader 없음")
        header = ole.openstream("FileHeader").read()
        compressed = bool(header[36] & 1)
        sections = sorted(
            (e for e in ole.listdir() if e[0] == "BodyText"),
            key=lambda e: int(e[1].replace("Section", "")),
        )
        if not sections:
            raise HwpParseError(f"{path.name}: BodyText 없음")
        parts: list[str] = []
        for entry in sections:
            data = ole.openstream(entry).read()
            if compressed:
                data = zlib.decompress(data, -15)
            i = 0
            while i + 4 <= len(data):
                (hdr,) = struct.unpack_from("<I", data, i)
                tag = hdr & 0x3FF
                size = (hdr >> 20) & 0xFFF
                i += 4
                if size == 0xFFF:
                    (size,) = struct.unpack_from("<I", data, i)
                    i += 4
                if tag == HWPTAG_PARA_TEXT:
                    parts.append(_para_text(data[i : i + size]))
                i += size
        text = "\n".join(parts)
        # chr() 단위 복원 과정에서 생긴 서러게이트 짝을 결합하고, 홀로 남은 것은 제거
        return text.encode("utf-16", "surrogatepass").decode("utf-16", errors="ignore")
    finally:
        ole.close()
