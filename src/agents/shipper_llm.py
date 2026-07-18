"""LLM-driven shipper carrier-selection.

This module replaces the legacy pure-pseudo implementation. A shipper is
backed by an `LLMBackend` (registry-keyed: openai / anthropic / google /
xai / mistral / cohere / ollama / pseudo). For each round the shipper:

  1. assembles a prompt with `build_shipper_prompt`,
  2. calls its backend (with response cache + retry),
  3. parses {"choice": rank, "reason": ...},
  4. converts the 1-based rank back to the carrier index.

Policy mixing (monoculture / polyculture / random) is configured at the
pool level via `ShipperPolicy`. This is what lets the experiment toggle
"all shippers use Claude" vs. "shippers are evenly split among GPT/Claude/
Gemini" vs. "60% GPT, 30% Claude, 10% Gemini".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..llm.base import LLMBackend, LLMRequestError
from ..llm.cache import ResponseCache
from ..llm.prompts import build_shipper_prompt
from ..llm.registry import make_backend
from .carriers import CarrierAttr


# Curated set of shipper "types" prompted to the LLM, drawn round-robin.
SHIPPER_TYPES = [
    "cost first (minimize freight cost)",
    "reliability first (avoid delivery problems)",
    "capacity first (prefer carriers that can handle volume)",
    "specialty first (prefer carriers whose expertise fits the route)",
    "balanced (weigh all factors evenly)",
]


# ---------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------
@dataclass
class LLMSpec:
    """One LLM 'kind' a shipper can use."""
    backend: str               # "openai" / "anthropic" / "google" / ...
    model: Optional[str] = None  # None -> registry default
    temperature: float = 0.7

    @property
    def label(self) -> str:
        m = self.model or "default"
        return f"{self.backend}:{m}"


@dataclass
class ShipperPolicy:
    """How LLM kinds are distributed across shippers.

    Examples:
        # monoculture (everyone is Claude Haiku):
        ShipperPolicy(specs=[LLMSpec("anthropic")])

        # polyculture (round-robin across 3 vendors):
        ShipperPolicy(
            specs=[LLMSpec("openai"), LLMSpec("anthropic"), LLMSpec("google")],
            mix_mode="polyculture",
        )

        # random 60/30/10:
        ShipperPolicy(
            specs=[LLMSpec("openai"), LLMSpec("anthropic"), LLMSpec("google")],
            mix_mode="random",
            weights=[0.6, 0.3, 0.1],
        )
    """
    specs: List[LLMSpec]
    mix_mode: str = "monoculture"     # "monoculture" / "polyculture" / "random"
    weights: Optional[List[float]] = None

    def __post_init__(self):
        if not self.specs:
            raise ValueError("ShipperPolicy requires at least one LLMSpec")
        if self.mix_mode == "random":
            if self.weights is None:
                self.weights = [1.0 / len(self.specs)] * len(self.specs)
            if len(self.weights) != len(self.specs):
                raise ValueError("weights length must match specs length")

    def assign_to_shippers(self, n_shippers: int,
                          rng: np.random.Generator) -> List[LLMSpec]:
        if self.mix_mode == "monoculture":
            return [self.specs[0]] * n_shippers
        if self.mix_mode == "polyculture":
            return [self.specs[i % len(self.specs)] for i in range(n_shippers)]
        if self.mix_mode == "random":
            idx = rng.choice(len(self.specs), size=n_shippers, p=self.weights)
            return [self.specs[int(i)] for i in idx]
        raise ValueError(f"unknown mix_mode {self.mix_mode!r}")


@dataclass
class ShipperConfig:
    """Per-cell configuration handed to ShipperPool.create()."""
    policy: ShipperPolicy
    shuffle_candidates: bool = True
    dynamic_profile: bool = False
    cache_path: Optional[str] = None     # None disables disk caching
    fallback_to_random_on_error: bool = True
    rating_mode: str = "endogenous"      # endogenous / static / truth


# ---------------------------------------------------------------------------
# Per-shipper agent
# ---------------------------------------------------------------------------
@dataclass
class ShipperLLM:
    shipper_idx: int
    shipper_type: str
    backend: LLMBackend
    config: ShipperConfig
    cache: Optional[ResponseCache] = None
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.shipper_idx * 1009 + 1)

    # ------------------------------------------------------------------
    def select(self, candidates: Sequence[Tuple[int, CarrierAttr]],
               carrier_degrees: np.ndarray,
               request_descr: str,
               history_summary: str,
               distance: Optional[int] = None,
               tons: Optional[int] = None,
               capacity_slots: Optional[Sequence[Optional[int]]] = None,
               capacity_left: Optional[Sequence[Optional[int]]] = None,
               ) -> Tuple[List[int], dict]:
        """Return (ranked_carrier_indices, metadata).

        ranked_carrier_indices is in preference order (up to three, no
        duplicates); waterfall tendering offers the load down this list.
        Metadata includes
        the LLM's free-text reason, cache hit flag and token usage.
        """
        prompt = build_shipper_prompt(
            shipper_idx=self.shipper_idx,
            shipper_type=self.shipper_type,
            request_descr=request_descr,
            history_summary=history_summary,
            candidates=candidates,
            carrier_degrees=carrier_degrees,
            dynamic_profile=self.config.dynamic_profile,
            distance=distance,
            tons=tons,
            capacity_slots=capacity_slots,
            capacity_left=capacity_left,
            rating_mode=self.config.rating_mode,
        )

        # Cache lookup (keyed on backend+model+prompt).
        cached: Optional["LLMResponse"] = None
        if self.cache is not None:
            cached = self.cache.get(self.backend.name, self.backend.model,
                                    self.backend.temperature, prompt)

        if cached is not None:
            resp = cached
        else:
            try:
                resp = self.backend.call(prompt)
            except LLMRequestError as exc:
                if self.config.fallback_to_random_on_error:
                    k = min(3, len(candidates))
                    picks = self.rng.choice(len(candidates), size=k,
                                            replace=False)
                    ranked = [candidates[int(p)][0] for p in picks]
                    return ranked, {"error": str(exc), "fallback": "random",
                                    "prompt": prompt, "raw_response": ""}
                raise
            if self.cache is not None:
                self.cache.put(resp, prompt, self.backend.temperature)

        # Interpret response numbers as carrier IDs and keep only those in
        # the candidate pool (the prompt asks for carrier_XX answers). Only
        # if no number resolves as an ID, fall back to interpreting them as
        # legacy display ranks 1..N.
        pool_ids = [j for j, _ in candidates]
        raw_choices = list(resp.choices or [resp.choice])
        ranked: List[int] = []
        for c in raw_choices:
            c = int(c)
            if c in pool_ids and c not in ranked:
                ranked.append(c)
            if len(ranked) >= 3:
                break
        if not ranked:
            for c in raw_choices:
                c = max(1, min(int(c), len(candidates)))
                j = candidates[c - 1][0]
                if j not in ranked:
                    ranked.append(j)
                if len(ranked) >= 3:
                    break
        return ranked, {
            "reason": resp.reason,
            "cached": resp.cached,
            "input_tokens": resp.usage_input_tokens,
            "output_tokens": resp.usage_output_tokens,
            "backend": resp.backend,
            "model": resp.model,
            # full I/O for tracing / audit
            "prompt": prompt,
            "raw_response": resp.raw_text,
            "choices_as_returned": list(resp.choices),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_shipper_pool(n_shippers: int, config: ShipperConfig,
                       rng: np.random.Generator) -> List[ShipperLLM]:
    """Materialise N shippers, each bound to one backend instance.

    Backend objects are instantiated *per unique spec* (not per shipper) so
    every shipper using the same provider shares one HTTP client.
    """
    assignments = config.policy.assign_to_shippers(n_shippers, rng)
    backend_cache: dict = {}

    def _get_backend(spec: LLMSpec) -> LLMBackend:
        key = spec.label
        if key not in backend_cache:
            backend_cache[key] = make_backend(
                f"{spec.backend}:{spec.model}" if spec.model else spec.backend,
                temperature=spec.temperature,
            )
        return backend_cache[key]

    response_cache: Optional[ResponseCache] = None
    if config.cache_path:
        from pathlib import Path
        response_cache = ResponseCache(Path(config.cache_path))

    shippers: List[ShipperLLM] = []
    for i in range(n_shippers):
        spec = assignments[i]
        shippers.append(ShipperLLM(
            shipper_idx=i,
            shipper_type=SHIPPER_TYPES[i % len(SHIPPER_TYPES)],
            backend=_get_backend(spec),
            config=config,
            cache=response_cache,
        ))
    return shippers


N_LOCATIONS = 20  # number of abstract locations (loc_00 .. loc_19)
TONS_CHOICES = (5, 10, 15, 20, 30)  # load tonnage levels (shared with economy's capacity calibration)


def synthesize_request(rng: np.random.Generator) -> Tuple[str, int, int]:
    """Generate one shipping request. Returns (description, distance, tons).

    Requests use abstract location ids and a distance index drawn
    uniformly from 10-100. Tonnage consumes carrier capacity (tons/day)
    proportionally, and money (quotes, revenue, cost) scales with
    tons x distance.
    """
    o = int(rng.integers(0, N_LOCATIONS))
    d = int(rng.integers(0, N_LOCATIONS - 1))
    if d >= o:
        d += 1  # origin and destination always differ
    distance = int(rng.integers(10, 101))
    tons = int(rng.choice(TONS_CHOICES))
    # Deadlines are made consistent with distance (no contradictory
    # combinations): distance 10-20 -> 1 day / >20-50 -> 2 / >50-100 -> 3
    if distance <= 20:
        days = 1
    elif distance <= 50:
        days = 2
    else:
        days = 3
    descr = (f"loc_{o:02d} -> loc_{d:02d}, distance {distance}, "
             f"{tons} tons, within {days} day{'s' if days > 1 else ''}")
    return descr, distance, tons
