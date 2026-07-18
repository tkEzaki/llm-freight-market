"""Baselines B1-B4 (random / pure PA / Leung-Weitz / score-greedy).

All baselines return the same ranked top-3 preference list (no
duplicates) as the LLM agents; waterfall tendering then offers the load
down the list through the same machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .carriers import CarrierAttr


def _k(candidates) -> int:
    return min(3, len(candidates))


def select_random(candidates: Sequence[Tuple[int, CarrierAttr]],
                  rng: np.random.Generator) -> List[int]:
    """B1 random: three carriers drawn uniformly, in preference order."""
    picks = rng.choice(len(candidates), size=_k(candidates), replace=False)
    return [candidates[int(p)][0] for p in picks]


def select_pure_pa(candidates: Sequence[Tuple[int, CarrierAttr]],
                   carrier_degrees: np.ndarray,
                   alpha: float = 1.0,
                   rng: np.random.Generator = None) -> List[int]:
    """B2 preferential attachment: draw 3 without replacement, P(j) ~ k_j + alpha."""
    rng = rng or np.random.default_rng()
    weights = np.array([carrier_degrees[j] + alpha for j, _ in candidates])
    weights = weights / weights.sum()
    picks = rng.choice(len(candidates), size=_k(candidates), replace=False,
                       p=weights)
    return [candidates[int(p)][0] for p in picks]


def select_conflicting_attachment(candidates: Sequence[Tuple[int, CarrierAttr]],
                                  carrier_degrees: np.ndarray,
                                  beta: float = 0.05,
                                  alpha: float = 1.0,
                                  rng: np.random.Generator = None) -> List[int]:
    """B3 conflicting attachment: P(j) ~ (k_j+alpha)*exp(-beta*k_j), 3 draws w/o replacement."""
    rng = rng or np.random.default_rng()
    raw = np.array([(carrier_degrees[j] + alpha) * np.exp(-beta * carrier_degrees[j])
                    for j, _ in candidates])
    raw = raw / raw.sum()
    picks = rng.choice(len(candidates), size=_k(candidates), replace=False,
                       p=raw)
    return [candidates[int(p)][0] for p in picks]


# Score-greedy uses a fixed common scoring vector (same for every shipper).
# This is the closest in-distribution proxy for Peng-Garg algorithmic
# monoculture where the decision rule is identical and deterministic.
# Four components matching the observed attribute vector (price, rating,
# capacity, specialty). Greedy sees the same observed rating as the LLM
# agents; the true reliability is hidden from everyone.
COMMON_SCORE = np.array([-0.8, 1.1, 0.8, 1.0])  # mid-mix


def select_score_greedy(candidates: Sequence[Tuple[int, CarrierAttr]]
                        ) -> List[int]:
    """B4 common-score: deterministic top-3 by a shared linear score (monoculture)."""
    scores = np.array([float(COMMON_SCORE @ attr.to_vector())
                       for _, attr in candidates])
    order = np.argsort(scores)[::-1][:_k(candidates)]
    return [candidates[int(i)][0] for i in order]
