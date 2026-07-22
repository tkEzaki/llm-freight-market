"""Anthropic / Claude-family backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import LLMBackend
from .registry import register


@register("anthropic")
@dataclass
class AnthropicBackend(LLMBackend):
    name: str = "anthropic"
    model: str = "claude-haiku-4-5"
    api_key_env: str = "ANTHROPIC_API_KEY"

    @classmethod
    def from_spec(cls, *, model: Optional[str], **kwargs) -> "AnthropicBackend":
        return cls(model=model or "claude-haiku-4-5", **kwargs)

    def _call(self, prompt: str) -> Dict[str, Any]:
        import os

        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ.get(self.api_key_env) or None)
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
            messages=[{"role": "user", "content": prompt}],
        )
        # Claude returns a list of content blocks; concat all text-type blocks.
        text = "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(resp, "usage", None)
        return {
            "text": text,
            "usage_input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "usage_output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        }
