"""Local Ollama backend (OSS models served on http://localhost:11434)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import LLMBackend
from .registry import register


@register("ollama")
@dataclass
class OllamaBackend(LLMBackend):
    name: str = "ollama"
    model: str = "llama3.1"
    # Ollama does not need a key; we still expose an env var so users can
    # 'unset' it to disable the backend in a sweep.
    api_key_env: str = "OLLAMA_ENABLE"   # set to "1" to mark as available
    host: str = "http://localhost:11434"

    @classmethod
    def from_spec(cls, *, model: Optional[str], **kwargs) -> "OllamaBackend":
        return cls(model=model or "llama3.1", **kwargs)

    def is_available(self) -> bool:
        return os.environ.get(self.api_key_env, "").strip() == "1"

    def _call(self, prompt: str) -> Dict[str, Any]:
        import requests
        url = f"{self.host.rstrip('/')}/api/generate"
        resp = requests.post(
            url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        return {
            "text": text,
            "usage_input_tokens": int(data.get("prompt_eval_count", 0)),
            "usage_output_tokens": int(data.get("eval_count", 0)),
        }
