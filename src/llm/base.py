"""Abstract LLM backend interface for shipper carrier-selection."""
from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# API keys can live in a .env file instead of real environment variables.
# Only this project's .env is loaded. Existing environment
# variables always take precedence (load_dotenv default).
try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # Only this project's .env is read; a parent directory's .env is
    # deliberately ignored so that keys from an unrelated project cannot
    # leak into these experiments.
    # utf-8-sig: tolerate the BOM added by Windows Notepad / PowerShell
    load_dotenv(_PROJECT_ROOT / ".env", encoding="utf-8-sig")
except ImportError:
    pass


class LLMRequestError(RuntimeError):
    """Raised when an LLM call ultimately fails after retries."""


class LLMQuotaError(RuntimeError):
    """Raised on non-retryable billing/quota errors (fail fast, no fallback).

    Billing/quota errors stop the whole experiment immediately so that
    random fallbacks never contaminate the data.
    """


_QUOTA_MARKERS = ("insufficient_quota", "billing", "exceeded your current quota")


@dataclass
class LLMResponse:
    """Parsed shipper-selection response from any LLM backend."""
    choice: int                          # 1-based rank of the FIRST preference
    reason: str                          # free-text rationale (may be empty)
    choices: List[int] = field(default_factory=list)  # ranked prefs (1-based)
    raw_text: str = ""                   # full response text (for logging)
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    backend: str = ""
    model: str = ""
    cached: bool = False


@dataclass
class LLMBackend(abc.ABC):
    """Common interface every backend implements."""
    name: str                # e.g. "openai"
    model: str               # e.g. "gpt-4o-mini"
    api_key_env: str         # name of env var holding the API key
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 30.0
    n_retries: int = 3
    backoff_seconds: float = 1.5

    # --- subclass extension points -------------------------------------
    @abc.abstractmethod
    def _call(self, prompt: str) -> Dict[str, Any]:
        """Provider-specific raw call. Must return:
            {"text": <str>, "usage_input_tokens": int,
             "usage_output_tokens": int}
        """

    # --- shared helpers ------------------------------------------------
    def is_available(self) -> bool:
        """True if the API key is configured in the environment."""
        return bool(os.environ.get(self.api_key_env, "").strip())

    def call(self, prompt: str) -> LLMResponse:
        """Public entry: handles retries, parsing, error wrapping."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.n_retries):
            try:
                result = self._call(prompt)
                text = result.get("text", "")
                parsed = parse_selection_response(text)
                choices = normalize_choices(parsed)
                return LLMResponse(
                    choice=choices[0],
                    choices=choices,
                    reason=parsed.get("reason", ""),
                    raw_text=text,
                    usage_input_tokens=int(result.get("usage_input_tokens", 0)),
                    usage_output_tokens=int(result.get("usage_output_tokens", 0)),
                    backend=self.name,
                    model=self.model,
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if any(m in msg for m in _QUOTA_MARKERS):
                    raise LLMQuotaError(
                        f"{self.name}:{self.model} quota/billing error "
                        f"(fail fast, no fallback): {exc}") from exc
                last_exc = exc
                if attempt < self.n_retries - 1:
                    time.sleep(self.backoff_seconds * (2 ** attempt))
        raise LLMRequestError(
            f"{self.name}:{self.model} failed after {self.n_retries} retries: {last_exc}"
        )


# ----------------------------------------------------------------------
# Response parser shared by every backend
# ----------------------------------------------------------------------
def parse_selection_response(text: str) -> Dict[str, Any]:
    """Extract {choice, reason} from a model's response text.

    Accepts strict JSON, JSON inside ```json fences, or a fallback search
    for the first integer in the text. Raises ValueError on hard failure.
    """
    import json
    import re

    candidates = []
    # 1) Direct JSON parse.
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) Fenced JSON block.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    # 3) Any {...} object substring.
    obj = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group(0))
        except Exception:
            pass
    # 4) Fallback: collect up to three carrier_XX tokens, then bare integers.
    ids = re.findall(r"carrier[_\s]?(\d{1,2})", text, flags=re.IGNORECASE)
    if not ids:
        ids = re.findall(r"\b([1-9]\d?)\b", text)
    if ids:
        seen: list = []
        for x in ids:
            v = int(x)
            if v not in seen:
                seen.append(v)
        return {"choices": seen[:3], "reason": text.strip()[:200]}
    raise ValueError(f"Could not parse selection from response: {text[:200]!r}")


def _coerce_choice(c: Any) -> Optional[int]:
    """'carrier_07' / 'carrier 7' / 7 / '07' -> 7; None if unparsable."""
    if isinstance(c, (int, float)):
        return int(c)
    if isinstance(c, str):
        import re as _re
        m = _re.search(r"(\d{1,3})", c)
        if m:
            return int(m.group(1))
    return None


def normalize_choices(parsed: Dict[str, Any]) -> List[int]:
    """Extract a deduplicated ranked-preference list from a parsed response.

    Elements are carrier IDs ("carrier_07" / 7); the legacy form
    {"choice": 2} is also accepted. Returns a list of ints (IDs, or display
    ranks for legacy responses — the caller disambiguates).
    Raises ValueError if nothing usable.
    """
    raw = parsed.get("choices")
    out: List[int] = []
    if isinstance(raw, (list, tuple)):
        for c in raw:
            v = _coerce_choice(c)
            if v is not None and v not in out:
                out.append(v)
    if not out and "choice" in parsed:
        v = _coerce_choice(parsed["choice"])
        if v is not None:
            out = [v]
    if not out:
        raise ValueError(f"no usable choice in parsed response: {parsed!r}")
    return out
