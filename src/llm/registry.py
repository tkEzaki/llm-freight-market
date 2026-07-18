"""Factory that turns a 'backend:model' string into an LLMBackend."""
from __future__ import annotations

from typing import Dict, List, Type

from .base import LLMBackend


# Maps short backend name → class. Populated lazily so importing this module
# does not require every provider SDK to be installed.
_REGISTRY: Dict[str, Type[LLMBackend]] = {}


def register(name: str):
    def deco(cls: Type[LLMBackend]) -> Type[LLMBackend]:
        _REGISTRY[name] = cls
        return cls
    return deco


def _ensure_registered() -> None:
    """Trigger lazy import of every shipped backend (idempotent)."""
    # The imports below register themselves with @register on import.
    from . import openai_backend     # noqa: F401
    from . import anthropic_backend  # noqa: F401
    from . import google_backend     # noqa: F401
    from . import xai_backend        # noqa: F401
    from . import mistral_backend    # noqa: F401
    from . import cohere_backend     # noqa: F401
    from . import ollama_backend     # noqa: F401
    from . import pseudo_backend     # noqa: F401


def available_backends() -> List[str]:
    _ensure_registered()
    return sorted(_REGISTRY.keys())


def make_backend(spec: str, *,
                 temperature: float = 0.7,
                 max_tokens: int = 256) -> LLMBackend:
    """Instantiate a backend from a 'name[:model]' spec string.

    Examples:
        make_backend("openai:gpt-4o-mini")
        make_backend("anthropic:claude-haiku-4-5")
        make_backend("google:gemini-2.5-flash")
        make_backend("pseudo:gpt")
    """
    _ensure_registered()
    if ":" in spec:
        name, model = spec.split(":", 1)
    else:
        name, model = spec, None
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Known: {sorted(_REGISTRY.keys())}")
    cls = _REGISTRY[name]
    return cls.from_spec(model=model, temperature=temperature,
                         max_tokens=max_tokens)


# Convenient curated default model per provider (used when spec has no model)
DEFAULT_MODELS: Dict[str, str] = {
    "openai":    "gpt-5.4-mini",
    "anthropic": "claude-haiku-4-5",
    "google":    "gemini-3.5-flash",
    "xai":       "grok-3-mini",
    "mistral":   "mistral-small-latest",
    "cohere":    "command-r",
    "ollama":    "llama3.1",
    "pseudo":    "gpt",
}
