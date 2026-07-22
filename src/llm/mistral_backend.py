"""Mistral AI backend (Le Chat family)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import LLMBackend
from .registry import register


@register("mistral")
@dataclass
class MistralBackend(LLMBackend):
    name: str = "mistral"
    model: str = "mistral-small-latest"
    api_key_env: str = "MISTRAL_API_KEY"

    @classmethod
    def from_spec(cls, *, model: Optional[str], **kwargs) -> "MistralBackend":
        return cls(model=model or "mistral-small-latest", **kwargs)

    def _call(self, prompt: str) -> Dict[str, Any]:
        import os

        from mistralai import Mistral

        client = Mistral(api_key=os.environ.get(self.api_key_env) or None)
        resp = client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return {
            "text": text,
            "usage_input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "usage_output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        }
