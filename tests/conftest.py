"""테스트 공통 설정.

운영 기본값은 deny by default(ADMIN_ENABLED=false)다. 관리 API 테스트는
관리용 인스턴스를 전제하므로 세션 전체에 활성 상태를 깔아준다.
비활성 동작을 검증하는 테스트는 개별적으로 False를 다시 지정한다.
(세션 스코프 — 모듈 스코프 앱 픽스처가 만들어지기 전에 적용되어야 라우트가 등록된다)
"""
import pytest

from app.config import settings


@pytest.fixture(autouse=True, scope="session")
def _admin_enabled_for_tests():
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "ADMIN_ENABLED", True)
    yield
    mp.undo()
