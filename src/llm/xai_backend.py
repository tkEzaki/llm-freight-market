"""xAI / Grok backend (uses the OpenAI-compatible endpoint)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .openai_backend import OpenAIBackend
from .registry import register


@register("xai")
@dataclass
class XAIBackend(OpenAIBackend):
    name: str = "xai"
    model: str = "grok-3-mini"
    api_key_env: str = "XAI_API_KEY"
    base_url: Optional[str] = "https://api.x.ai/v1"
    force_json_object: bool = False  # not all Grok models accept response_format

    @classmethod
    def from_spec(cls, *, model: Optional[str], **kwargs) -> "XAIBackend":
        return cls(model=model or "grok-3-mini", **kwargs)

    # No _call override: OpenAIBackend reads the key from api_key_env,
    # which is XAI_API_KEY here, and posts it to base_url.
