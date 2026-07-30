from app.config import settings


def test_settings_paths():
    assert settings.GEN_MODEL == "hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:Q4_K_M"
    assert settings.CORPUS_DIR.name == "corpus"
    assert settings.RELEVANCE_THRESHOLD == 0.35
