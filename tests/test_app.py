from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from social_editor import config

APP = str(Path(config.__file__).with_name("app.py"))


@pytest.fixture
def no_keys(monkeypatch):
    # Empty values (rather than unset) also stop load_dotenv() reading a local .env.
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "MODEL"):
        monkeypatch.setenv(name, "")


def test_default_model_needs_a_gemini_key(no_keys):
    assert config.model_name() == config.DEFAULT_MODEL
    assert "GEMINI_API_KEY or GOOGLE_API_KEY" in config.missing_key_error(config.model_name())


def test_either_google_key_is_enough(no_keys, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert config.missing_key_error("gemini/gemini-3.8-flash") is None


def test_other_providers_are_checked_or_passed_through(no_keys):
    assert "OPENAI_API_KEY" in config.missing_key_error("openai/gpt-5")
    assert config.missing_key_error("ollama/llama3") is None


def test_app_explains_a_missing_key(no_keys):
    at = AppTest.from_file(APP).run()
    assert "GEMINI_API_KEY" in at.error[0].value
    assert not at.text_area


def test_app_shows_each_post_with_its_critique(no_keys, monkeypatch, llm):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(config, "make_llm", lambda model: llm)

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area[0].input("We're hiring a designer")
    at.button[0].click().run()

    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Instagram", "TikTok", "LinkedIn"]
    assert at.code[0].value == "Final Instagram post\n\n#Logo #Design"
    assert "Lead with the prize" in at.expander[0].markdown[0].value
