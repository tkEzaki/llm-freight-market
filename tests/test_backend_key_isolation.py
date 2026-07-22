# -*- coding: utf-8 -*-
"""Each backend must send only its own API key, to its own endpoint.

The failure this guards against: the xAI backend reuses the OpenAI SDK
against https://api.x.ai/v1, so if the SDK is allowed to pick the key up
from OPENAI_API_KEY, an OpenAI key would be transmitted to xAI (and vice
versa). Run with:  python -m pytest tests -q
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.anthropic_backend import AnthropicBackend  # noqa: E402
from src.llm.openai_backend import OpenAIBackend        # noqa: E402
from src.llm.xai_backend import XAIBackend              # noqa: E402

ALL_KEY_VARS = ["OPENAI_API_KEY", "XAI_API_KEY", "ANTHROPIC_API_KEY",
                "MISTRAL_API_KEY", "COHERE_API_KEY", "GOOGLE_API_KEY"]
JSON_REPLY = '{"choices": ["carrier_01"], "reason": "test"}'


@pytest.fixture(autouse=True)
def all_keys_set(monkeypatch):
    """Every vendor key is present: the situation that leaked before."""
    for var in ALL_KEY_VARS:
        monkeypatch.setenv(var, f"secret-for-{var}")


def _fake_openai_sdk(seen):
    """Minimal stand-in for the openai package that records what it gets."""
    class FakeCompletions:
        def create(self, **kw):
            msg = types.SimpleNamespace(content=JSON_REPLY)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg)],
                usage=types.SimpleNamespace(prompt_tokens=1,
                                            completion_tokens=1))

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, **kw):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    mod = types.ModuleType("openai")
    mod.OpenAI = FakeOpenAI
    return mod


def test_openai_backend_uses_only_openai_key(monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_openai_sdk(seen))
    OpenAIBackend()._call("prompt")
    assert seen["api_key"] == "secret-for-OPENAI_API_KEY"
    assert seen["base_url"] is None


def test_xai_backend_never_sends_the_openai_key(monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_openai_sdk(seen))
    XAIBackend()._call("prompt")
    assert seen["api_key"] == "secret-for-XAI_API_KEY"
    assert seen["api_key"] != "secret-for-OPENAI_API_KEY"
    assert seen["base_url"] == "https://api.x.ai/v1"


def test_xai_backend_does_not_mutate_the_process_environment(monkeypatch):
    import os
    seen = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_openai_sdk(seen))
    XAIBackend()._call("prompt")
    assert os.environ["OPENAI_API_KEY"] == "secret-for-OPENAI_API_KEY"


def test_xai_backend_does_not_populate_an_unset_openai_key(monkeypatch):
    """With no OpenAI key configured, running xAI must not create one.

    Otherwise the xAI key would be left in OPENAI_API_KEY and a later
    OpenAI call in the same process would ship it to OpenAI.
    """
    import os
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_openai_sdk(seen))
    XAIBackend()._call("prompt")
    assert seen["api_key"] == "secret-for-XAI_API_KEY"
    assert "OPENAI_API_KEY" not in os.environ


def test_anthropic_backend_uses_only_anthropic_key(monkeypatch):
    seen = {}

    class FakeMessages:
        def create(self, **kw):
            block = types.SimpleNamespace(type="text", text=JSON_REPLY)
            return types.SimpleNamespace(
                content=[block],
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))

    class FakeAnthropic:
        def __init__(self, api_key=None, **kw):
            seen["api_key"] = api_key
            self.messages = FakeMessages()

    mod = types.ModuleType("anthropic")
    mod.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    AnthropicBackend()._call("prompt")
    assert seen["api_key"] == "secret-for-ANTHROPIC_API_KEY"


def test_no_backend_reads_a_parent_directory_dotenv():
    """base.py must load this project's .env only."""
    src = (Path(__file__).resolve().parents[1] / "src" / "llm"
           / "base.py").read_text(encoding="utf-8")
    assert ".parent / \".env\"" not in src
