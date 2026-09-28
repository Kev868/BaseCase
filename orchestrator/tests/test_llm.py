"""Which model the agent runs on, and with which key.

Building a LangChain model does not call it, so nothing here reaches a
network. `conftest` blanks every real key; each test supplies its own.
"""

from __future__ import annotations

import pytest

from agent.llm import NoModelConfigured, build_model
from config import XAI_BASE_URL, Config, provider_of
from memory.store import PhraseMemory


def test_grok_runs_through_the_same_client_at_xais_address():
    model = build_model("xai", "grok-4.7", "xai-test")
    assert model.model_name == "grok-4.7"
    assert model.openai_api_base == XAI_BASE_URL


def test_openai_keeps_its_own_address():
    model = build_model("openai", "gpt-6-astra", "sk-test")
    assert model.openai_api_base is None


def test_a_missing_key_names_the_provider_it_is_missing_for():
    with pytest.raises(NoModelConfigured, match="'xai'"):
        build_model("xai", "grok-4.7", "")


@pytest.mark.parametrize("model, provider", [
    ("grok-4.7", "xai"),
    ("grok-build-0.1", "xai"),
    ("gpt-6-astra", "openai"),
    ("claude-opus-5-5", "anthropic"),
])
def test_the_provider_follows_the_model_name(model, provider):
    assert provider_of(model) == provider


def test_each_provider_reads_its_own_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("XAI_API_KEY", "xai-grok")
    assert Config(provider="openai").api_key == "sk-openai"
    assert Config(provider="xai").api_key == "xai-grok"


def test_the_phrase_memory_embeds_with_openai_even_when_the_agent_is_on_grok(
        monkeypatch, tmp_path):
    """The embedding models are OpenAI's. Handing them the Grok key would be
    refused on every lookup, and the memory would quietly stop recalling."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("XAI_API_KEY", "xai-grok")
    monkeypatch.setattr("memory.store.CFG", Config(provider="xai"))

    memory = PhraseMemory(uri=str(tmp_path / "lancedb"), embed_model="")
    memory.embed_model = "text-embedding-3-small"
    embedder = memory._make_embedder()

    assert embedder.openai_api_key.get_secret_value() == "sk-openai"
