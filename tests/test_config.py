from app.config import settings


def test_settings_paths():
    assert settings.GEN_MODEL == "exaone3.5:2.4b"
    assert settings.CORPUS_DIR.name == "corpus"
    assert settings.RELEVANCE_THRESHOLD == 0.35
