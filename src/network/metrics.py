"""Network observables for RQ1 (§6 of model design)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def gini_coefficient(values: np.ndarray) -> float:
    """Gini of a non-negative vector. Returns 0 for all-zero input."""
    v = np.asarray(values, dtype=float).flatten()
    if v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    # Gini formula via Lorenz curve.
    return float((2 * np.sum((np.arange(1, n + 1)) * v) - (n + 1) * cum[-1]) / (n * cum[-1]))


def degree_distribution(degrees: np.ndarray, n_bins: int = 20) -> Dict[str, np.ndarray]:
    """Histogram + summary for carrier-degree distribution."""
    d = np.asarray(degrees, dtype=float)
    counts, edges = np.histogram(d, bins=n_bins)
    return {
        "counts": counts,
        "edges": edges,
        "max": float(d.max()) if d.size else 0.0,
        "mean": float(d.mean()) if d.size else 0.0,
        "std": float(d.std()) if d.size else 0.0,
    }


def condensation_ratio(degrees: np.ndarray, top_k: int = 3) -> float:
    """Fraction of edge mass held by the top-k carriers (κ_t).

    The headline indicator for hub concentration (§6.1 model spec).
    """
    d = np.asarray(degrees, dtype=float)
    total = d.sum()
    if total <= 0:
        return 0.0
    top = np.sort(d)[-top_k:]
    return float(top.sum() / total)


def edge_entropy(weights: np.ndarray) -> float:
    """Shannon entropy of the edge-weight distribution."""
    flat = weights.flatten()
    total = flat.sum()
    if total <= 0:
        return 0.0
    p = flat[flat > 0] / total
    return float(-np.sum(p * np.log(p)))


def rewiring_rate(prev_edges: set, curr_edges: set) -> float:
    """|E_t △ E_{t-1}| / max(|E_{t-1}|, 1)."""
    if not prev_edges:
        return float(len(curr_edges) > 0)
    sym_diff = prev_edges.symmetric_difference(curr_edges)
    return len(sym_diff) / max(len(prev_edges), 1)


def bipartite_modularity_proxy(weights: np.ndarray, n_communities: int = 4) -> float:
    """Cheap modularity proxy via k-means on carrier columns.

    The bipartite Newman-Leicht modularity is expensive to wire up cleanly for
    only this prototype. We approximate community structure by clustering
    carriers based on their incoming weight patterns and reporting the
    fraction of total edge mass that stays inside the largest community.
    """
    if weights.sum() == 0:
        return 0.0
    n_carriers = weights.shape[1]
    if n_carriers < n_communities:
        return 1.0
    # Assign each carrier to one of n_communities clusters by argmax over
    # quantile bins of its column sum (toy but stable, no sklearn dep).
    col_sums = weights.sum(axis=0)
    bins = np.quantile(col_sums, np.linspace(0, 1, n_communities + 1)[1:-1])
    assignments = np.searchsorted(bins, col_sums)
    # Largest-community share of total weight.
    largest = 0.0
    for c in range(n_communities):
        mass = weights[:, assignments == c].sum()
        if mass > largest:
            largest = mass
    return float(largest / weights.sum())


def snapshot(graph, prev_edges: Optional[set] = None) -> Dict[str, float]:
    """Convenience: all headline metrics at one timestep."""
    degrees = graph.carrier_degree()
    curr_edges = graph.edge_set()
    return {
        "kappa_top3": condensation_ratio(degrees, top_k=3),
        "kappa_top5": condensation_ratio(degrees, top_k=5),
        "gini": gini_coefficient(degrees),
        "entropy": edge_entropy(graph.weights),
        "modularity_proxy": bipartite_modularity_proxy(graph.weights),
        "rewiring": rewiring_rate(prev_edges or set(), curr_edges),
        "max_degree": float(degrees.max()) if degrees.size else 0.0,
        "n_active_edges": float(len(curr_edges)),
    }
