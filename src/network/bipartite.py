"""Weighted bipartite shipper-carrier graph with edge decay and pruning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class BipartiteGraph:
    n_shippers: int
    n_carriers: int
    decay: float = 0.95
    prune_threshold: float = 0.05
    # weights[i, j] = current weight of edge shipper_i <-> carrier_j
    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.weights = np.zeros((self.n_shippers, self.n_carriers), dtype=float)

    # ------------------------------------------------------------------
    def add_match(self, shipper_idx: int, carrier_idx: int, amount: float = 1.0) -> None:
        self.weights[shipper_idx, carrier_idx] += amount

    def decay_and_prune(self) -> None:
        self.weights *= self.decay
        # Anything below threshold is removed (zeroed).
        self.weights[self.weights < self.prune_threshold] = 0.0

    # ------------------------------------------------------------------
    def carrier_degree(self) -> np.ndarray:
        """Cumulative weight per carrier (k_j)."""
        return self.weights.sum(axis=0)

    def edge_set(self) -> set:
        """Active edges (weight > 0)."""
        rows, cols = np.nonzero(self.weights)
        return set(zip(rows.tolist(), cols.tolist()))

    def total_weight(self) -> float:
        return float(self.weights.sum())

    def shipper_choice_history(self, shipper_idx: int) -> Dict[int, float]:
        """Per-carrier weight viewed from a single shipper."""
        return {j: float(self.weights[shipper_idx, j])
                for j in range(self.n_carriers)
                if self.weights[shipper_idx, j] > 0}
