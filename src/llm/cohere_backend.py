"""Cohere Command-family backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import LLMBackend
from .registry import register


@register("cohere")
@dataclass
class CohereBackend(LLMBackend):
    name: str = "cohere"
    model: str = "command-r"
    api_key_env: str = "COHERE_API_KEY"

    @classmethod
    def from_spec(cls, *, model: Optional[str], **kwargs) -> "CohereBackend":
        return cls(model=model or "command-r", **kwargs)

    def _call(self, prompt: str) -> Dict[str, Any]:
        import os

        import cohere

        client = cohere.ClientV2(
            api_key=os.environ.get(self.api_key_env) or None)
        resp = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        # Cohere V2 returns: resp.message.content[0].text
        try:
            text = resp.message.content[0].text
        except Exception:
            text = str(resp)
        usage = getattr(resp, "usage", None)
        billed = getattr(usage, "billed_units", None) if usage else None
        return {
            "text": text,
            "usage_input_tokens": int(getattr(billed, "input_tokens", 0) or 0) if billed else 0,
            "usage_output_tokens": int(getattr(billed, "output_tokens", 0) or 0) if billed else 0,
        }
