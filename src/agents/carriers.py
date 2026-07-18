"""Carrier population: attributes phi_j + dynamic profile reinforcement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CarrierAttr:
    """Static / slowly-evolving carrier attributes used by all agents.

    Information structure:
      * reliability is the true quality — it only generates satisfaction
        draws and is never displayed.
      * Shippers (LLM/greedy/pseudo) see the experience rating (= satisfied
        transactions / n), written back by Economy at the end of each round.
      * base_profile_strength feeds the initial track record n0
        (heavy_tail: only a few incumbents start with deep records). It is
        never shown in prompts.
    """
    price: float          # lower = cheaper
    reliability: float    # 0..1  true quality (hidden; outcome generation only)
    capacity: float       # 0..1 (normalised)
    specialty: float      # 0..1 (route-domain match)
    base_profile_strength: float  # source of the track record n0 (fitness axis)
    rating: float = 0.5   # observed rating (written back by Economy; n=0 -> prior 0.5)
    rating_n: int = 0     # cumulative rated transactions (track-record depth)

    def to_vector(self) -> np.ndarray:
        """Only shipper-visible information (excludes true reliability/profile)."""
        return np.array([self.price, self.rating, self.capacity,
                         self.specialty])


def sample_carriers(m: int, rng: np.random.Generator,
                    fitness_regime: str = "uniform") -> List[CarrierAttr]:
    """Create M carriers with one of three fitness regimes (§7).

    fitness_regime ∈ {"uniform", "moderate", "heavy_tail"}.
    The 'fitness' here is concentrated in base_profile_strength to make
    monoculture & PA effects easy to read off the same axis.
    """
    if fitness_regime == "uniform":
        fitness = rng.uniform(0.4, 0.6, size=m)
    elif fitness_regime == "moderate":
        fitness = rng.normal(loc=0.5, scale=0.15, size=m)
        fitness = np.clip(fitness, 0.05, 0.95)
    elif fitness_regime == "heavy_tail":
        # Pareto-like: most carriers ordinary, a few outstanding
        raw = rng.pareto(a=1.5, size=m) + 0.1
        fitness = raw / raw.max() * 0.9 + 0.05
    else:
        raise ValueError(f"unknown fitness_regime: {fitness_regime}")

    return [
        CarrierAttr(
            price=float(rng.uniform(0.3, 0.9)),
            reliability=float(rng.uniform(0.3, 0.9)),
            capacity=float(rng.uniform(0.3, 0.9)),
            specialty=float(rng.uniform(0.3, 0.9)),
            base_profile_strength=float(fitness[j]),
        )
        for j in range(m)
    ]


def reinforced_profile_strength(attr: CarrierAttr, carrier_degree: float,
                                gain: float = 0.05) -> float:
    """When dynamic reinforcement is ON, popular carriers look better."""
    return min(1.0, attr.base_profile_strength + gain * np.log1p(carrier_degree))
