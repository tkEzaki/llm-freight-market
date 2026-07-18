"""Google / Gemini-family backend (google-genai SDK)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import LLMBackend
from .registry import register


@register("google")
@dataclass
class GoogleBackend(LLMBackend):
    name: str = "google"
    model: str = "gemini-3.5-flash"
    api_key_env: str = "GOOGLE_API_KEY"

    @classmethod
    def from_spec(cls, *, model: Optional[str], **kwargs) -> "GoogleBackend":
        return cls(model=model or "gemini-3.5-flash", **kwargs)

    def _call(self, prompt: str) -> Dict[str, Any]:
        # Prefer the newer 'google-genai' SDK; fall back to the older
        # 'google-generativeai' if needed.
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
            client = genai.Client(api_key=os.environ[self.api_key_env])
            # Gemini 2.5+ enables thinking by default, which can consume
            # max_output_tokens and leave the text empty; disable it, and
            # fall back to a config without thinking for models that do
            # not accept thinking_budget=0.
            base_cfg = dict(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            )
            try:
                resp = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        **base_cfg,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
            except Exception as exc:
                if "thinking" not in str(exc).lower():
                    raise
                resp = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**base_cfg),
                )
            text = getattr(resp, "text", "") or ""
            usage = getattr(resp, "usage_metadata", None)
            return {
                "text": text,
                "usage_input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "usage_output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
            }
        except ImportError:
            import google.generativeai as genai_old  # type: ignore
            genai_old.configure(api_key=os.environ[self.api_key_env])
            model = genai_old.GenerativeModel(self.model)
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            text = getattr(resp, "text", "") or ""
            usage = getattr(resp, "usage_metadata", None)
            return {
                "text": text,
                "usage_input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "usage_output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
            }
