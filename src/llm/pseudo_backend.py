"""Pseudo-LLM backend: deterministic scoring with knobs for the same
behavioural traits we wire up to real models (homogeneity, position bias,
self-reinforcement). Used for offline dry-runs without an API key.

The model name is interpreted as the 'family' (gpt / claude / llama / qwen).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .base import LLMBackend
from .registry import register


# Per-family weight vectors over (price, reliability, capacity, specialty,
# profile_strength). Mirrors src/agents/shipper_llm.LLM_FAMILY_WEIGHTS so a
# real-LLM run with the same prompt is comparable to the pseudo run.
FAMILY_WEIGHTS = {
    "gpt":    np.array([-0.7, 1.3, 0.6, 0.6, 1.4]),
    "claude": np.array([-0.9, 1.1, 0.7, 0.8, 0.9]),
    "llama":  np.array([-0.6, 0.8, 1.0, 1.2, 0.7]),
    "qwen":   np.array([-0.8, 0.9, 0.9, 1.0, 1.1]),
}


@register("pseudo")
@dataclass
class PseudoBackend(LLMBackend):
    name: str = "pseudo"
    model: str = "gpt"               # family
    api_key_env: str = "_PSEUDO_ALWAYS_AVAILABLE"
    rng_seed_env: str = "PSEUDO_SEED"

    @classmethod
    def from_spec(cls, *, model: Optional[str], **kwargs) -> "PseudoBackend":
        return cls(model=model or "gpt", **kwargs)

    def is_available(self) -> bool:
        return True

    # The pseudo backend parses attribute values straight out of the
    # rendered prompt so it can score candidates deterministically.
    # Attributes are extracted line by line so parsing is robust to
    # format variants (with/without carrier IDs, company-size units).
    _CAND_RE = re.compile(r"^\s*(\d+)\)\s+(?:carrier_(\d+):\s*)?(.+)$",
                          re.MULTILINE)
    _MAX_SLOTS_FOR_NORM = 5.0   # legacy 'n jobs/day' -> 0-1 normalizer
    _MAX_TONS_FOR_NORM = 80.0   # 'n tons/day' -> 0-1 normalizer

    def _parse_attrs(self, body: str):
        """Extract the 5 attributes, accepting current English and legacy Japanese labels."""
        import re as _re
        vals = []
        # Price: try 'unit price' (quote form), then 'price index' (raw).
        m = _re.search(r"unit price\s+([\d.]+)", body) \
            or _re.search(r"price index\s+([\d.]+)", body)
        if not m:
            return None
        vals.append(float(m.group(1)))
        # Trust signal: try 'customer rating', then 'reliability' (truth
        # mode / legacy). Unrated entrants get the uninformative prior 0.5.
        if _re.search(r"no customer ratings yet", body):
            vals.append(0.5)
        else:
            m = _re.search(r"customer rating\s+([\d.]+)", body) \
                or _re.search(r"reliability\s+([\d.]+)", body)
            if not m:
                return None
            vals.append(float(m.group(1)))
        # Company size: try 'up to n tons/day' (current), then
        # 'n jobs/day' (legacy), then a numeric score; normalize to 0-1.
        m = _re.search(r"up to (\d+) tons/day", body)
        if m:
            vals.append(min(1.0, int(m.group(1)) / self._MAX_TONS_FOR_NORM))
        else:
            m = _re.search(r"up to (\d+) jobs/day", body)
            if m:
                vals.append(min(1.0,
                                int(m.group(1)) / self._MAX_SLOTS_FOR_NORM))
            else:
                m = _re.search(r"company size\s+([\d.]+)", body)
                if not m:
                    return None
                vals.append(float(m.group(1)))
        m = _re.search(r"specialty\s+([\d.]+)", body)
        if not m:
            return None
        vals.append(float(m.group(1)))
        # 'profile quality' no longer exists in current prompts; it only
        # appears in legacy prompts (cache compatibility), so use it if
        # present, else 0 (weight x 0 disables it).
        m = _re.search(r"profile quality\s+([\d.]+)", body)
        vals.append(float(m.group(1)) if m else 0.0)
        return np.array(vals)

    def _call(self, prompt: str) -> Dict[str, Any]:
        family = self.model if self.model in FAMILY_WEIGHTS else "gpt"
        w = FAMILY_WEIGHTS[family]

        scores = []
        for m in self._CAND_RE.finditer(prompt):
            attrs = self._parse_attrs(m.group(3))
            if attrs is None:
                continue
            # Answer with the carrier ID when present (current prompt
            # spec), else with the display rank (legacy compatibility).
            label = (f"carrier_{int(m.group(2)):02d}" if m.group(2)
                     else int(m.group(1)))
            scores.append((label, float(w @ attrs)))

        if not scores:
            # If parsing failed, just pick rank 1.
            return {"text": json.dumps({"choice": 1, "reason": "fallback"}),
                    "usage_input_tokens": 0, "usage_output_tokens": 0}

        # Softmax-sampled TOP-3 ranking (without replacement) so pseudo runs
        # exercise the same waterfall-tendering path as real LLMs.
        rng_seed = os.environ.get(self.rng_seed_env)
        rng = np.random.default_rng(int(rng_seed) if rng_seed else None)
        s = np.array([v for _, v in scores]) / max(self.temperature, 1e-3)
        s -= s.max()
        probs = np.exp(s); probs /= probs.sum()
        k = min(3, len(scores))
        idxs = rng.choice(len(scores), size=k, replace=False, p=probs)
        ranks = [scores[int(i)][0] for i in idxs]
        return {
            "text": json.dumps({"choices": ranks,
                                "reason": f"pseudo family={family}"}),
            "usage_input_tokens": len(prompt) // 4,
            "usage_output_tokens": 25,
        }
