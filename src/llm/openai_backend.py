"""OpenAI / ChatGPT-family backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import LLMBackend
from .registry import register


@register("openai")
@dataclass
class OpenAIBackend(LLMBackend):
    name: str = "openai"
    model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None       # for Azure / proxy installs
    force_json_object: bool = True

    @classmethod
    def from_spec(cls, *, model: Optional[str], **kwargs) -> "OpenAIBackend":
        return cls(model=model or "gpt-5.4-mini", **kwargs)

    def _call(self, prompt: str) -> Dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url) if self.base_url else OpenAI()
        # GPT-5-family models require max_completion_tokens instead of
        # max_tokens and accept reasoning_effort='none' for non-thinking
        # behavior (parity with the other vendors). If an older model or an
        # OpenAI-compatible endpoint (e.g. xAI) rejects these parameters,
        # inspect the error message, adjust, and retry.
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            reasoning_effort="none",
            timeout=self.timeout,
        )
        if self.force_json_object:
            kwargs["response_format"] = {"type": "json_object"}

        resp = None
        for _ in range(4):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                msg = str(exc)
                if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort")
                elif "max_completion_tokens" in msg \
                        and "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                elif "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                elif "response_format" in msg and "response_format" in kwargs:
                    kwargs.pop("response_format")
                else:
                    raise
        if resp is None:
            raise RuntimeError(f"{self.name}:{self.model} parameter "
                               f"negotiation failed")
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return {
            "text": text,
            "usage_input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "usage_output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        }
