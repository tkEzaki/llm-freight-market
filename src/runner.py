"""Run one RQ1 simulation cell and return per-round observables.

New in this version: shipper LLM selection is done via real provider
backends (OpenAI / Anthropic / Google / xAI / Mistral / Cohere / Ollama)
or the offline pseudo backend, configured through `CellConfig.llm_specs`
and `llm_mix_mode` instead of the old `family_mix` / `families` knobs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .agents.baselines import (
    select_conflicting_attachment, select_pure_pa, select_random,
    select_score_greedy,
)
from .agents.carriers import CarrierAttr, sample_carriers
from .agents.shipper_llm import (
    LLMSpec, ShipperConfig, ShipperPolicy, build_shipper_pool,
    synthesize_request,
)
from .economy import Economy, EconomyConfig
from .network.bipartite import BipartiteGraph
from .network.metrics import condensation_ratio, snapshot


@dataclass
class CellConfig:
    """One simulation cell."""
    name: str
    selector: str = "llm"                # 'llm' / 'random' / 'pa' / 'conflicting' / 'greedy'
    n_shippers: int = 50
    n_carriers: int = 20
    n_rounds: int = 30
    candidate_pool_size: int = 5
    fitness_regime: str = "moderate"     # uniform / moderate / heavy_tail
    shuffle_candidates: bool = True
    dynamic_profile: bool = False
    decay: float = 0.95
    prune_threshold: float = 0.05
    seed: int = 0

    # ---- LLM-specific ------------------------------------------------
    # Examples:
    #   ["openai:gpt-4o-mini"]                           # monoculture
    #   ["openai", "anthropic", "google"]                # polyculture (round-robin)
    #   ["openai", "anthropic"]  + weights=[0.6, 0.4]   # random mix
    llm_specs: Tuple[str, ...] = ("pseudo:gpt",)
    llm_mix_mode: str = "monoculture"     # monoculture / polyculture / random
    llm_weights: Optional[Tuple[float, ...]] = None
    llm_temperature: float = 0.7
    cache_path: Optional[str] = None      # disk JSONL cache of LLM responses
    history_summary_length: int = 5       # last K choices kept in prompt

    # ---- Economy (P0: capacity / dynamic pricing / welfare) ----------
    econ_enabled: bool = True
    capacity_scale: float = 1.3          # system slots = scale × n_shippers
    price_adjust_rate: float = 0.10
    exit_enabled: bool = False
    # Real-time capacity disclosure (intervention arm). Default: hidden,
    # as in tendering practice — availability is unknown until tendered.
    show_capacity: bool = False
    # Trust-signal structure:
    #   endogenous = experience rating, updated by satisfaction draws on
    #                each completed transaction [default]
    #   static     = initial rating (n0 draws) never updated (feedback-severed control)
    #   truth      = true reliability disclosed (full-information control)
    rating_mode: str = "endogenous"
    rating_n0_scale: int = 200
    rating_n0_offset: int = 10
    # Full audit log (JSONL). A decision row holds one decision's prompt,
    # raw response, parsed ranking, and outcome; a round_state row holds
    # every carrier's state and all observables at the end of each round.
    trace_path: Optional[str] = None
    # LLM-call concurrency within a round. Prompts depend only on the
    # start-of-round state, so splitting into "parallel calls, then serve
    # sequentially in the original random order" leaves the semantics
    # unchanged. Automatically drops to 1 when show_capacity=True (the
    # disclosed remaining capacity is order-dependent within the day).
    llm_concurrency: int = 16

    def economy_config(self) -> EconomyConfig:
        return EconomyConfig(
            enabled=self.econ_enabled,
            capacity_scale=self.capacity_scale,
            price_adjust_rate=self.price_adjust_rate,
            exit_enabled=self.exit_enabled,
            rating_mode=self.rating_mode,
            rating_n0_scale=self.rating_n0_scale,
            rating_n0_offset=self.rating_n0_offset,
        )


def _build_candidate_pool(rng: np.random.Generator,
                          all_carriers: List[CarrierAttr],
                          pool_size: int,
                          shuffle: bool,
                          active_indices: Optional[List[int]] = None,
                          ) -> List[Tuple[int, CarrierAttr]]:
    idx_pool = active_indices if active_indices is not None \
        else list(range(len(all_carriers)))
    k = min(pool_size, len(idx_pool))
    chosen = rng.choice(idx_pool, size=k, replace=False)
    if shuffle:
        rng.shuffle(chosen)
    return [(int(j), all_carriers[int(j)]) for j in chosen]


def _summarize_history(history: List[Tuple[int, bool]], k: int) -> str:
    """Render the last k (choice, served) records into one prompt line.

    Failed loads (all three ranked carriers full) are flagged as
    "N failed" so the LLM can learn to avoid congested carriers.
    """
    if not history:
        return ""
    tail = history[-k:]
    counts: Dict[int, int] = {}
    fails: Dict[int, int] = {}
    for j, ok in tail:
        counts[j] = counts.get(j, 0) + 1
        if not ok:
            fails[j] = fails.get(j, 0) + 1
    parts = []
    for j, c in counts.items():
        note = f" ({fails[j]} failed)" if fails.get(j) else ""
        parts.append(f"carrier_{j:02d} x{c}{note}")
    return "last {} shipments: ".format(len(tail)) + ", ".join(parts)


def _make_shipper_pool(cfg: CellConfig, rng: np.random.Generator):
    specs = [_parse_spec(s, cfg.llm_temperature) for s in cfg.llm_specs]
    policy = ShipperPolicy(
        specs=specs,
        mix_mode=cfg.llm_mix_mode,
        weights=list(cfg.llm_weights) if cfg.llm_weights else None,
    )
    shipper_cfg = ShipperConfig(
        policy=policy,
        shuffle_candidates=cfg.shuffle_candidates,
        dynamic_profile=cfg.dynamic_profile,
        cache_path=cfg.cache_path,
        rating_mode=cfg.rating_mode,
    )
    return build_shipper_pool(cfg.n_shippers, shipper_cfg, rng)


def _parse_spec(spec_str: str, temperature: float) -> LLMSpec:
    if ":" in spec_str:
        backend, model = spec_str.split(":", 1)
    else:
        backend, model = spec_str, None
    return LLMSpec(backend=backend, model=model, temperature=temperature)


# ---------------------------------------------------------------------
def run_cell(cfg: CellConfig) -> pd.DataFrame:
    """Run one configuration; return per-round metrics dataframe."""
    rng = np.random.default_rng(cfg.seed)
    carriers = sample_carriers(cfg.n_carriers, rng, cfg.fitness_regime)
    graph = BipartiteGraph(cfg.n_shippers, cfg.n_carriers,
                           decay=cfg.decay, prune_threshold=cfg.prune_threshold)
    # Preference graph: records first choices regardless of whether they
    # were served; its gap to the realized graph is the structural version
    # of congestion displacement.
    pref_graph = BipartiteGraph(cfg.n_shippers, cfg.n_carriers,
                                decay=cfg.decay,
                                prune_threshold=cfg.prune_threshold)
    # Satisfaction draws use a dedicated rng (keeps the main stream's
    # consumption sequence, and thus seed reproducibility, unchanged).
    economy = Economy(carriers, cfg.n_shippers, cfg.economy_config(),
                      rng=np.random.default_rng(cfg.seed * 1_000_003 + 17))

    shippers = _make_shipper_pool(cfg, rng) if cfg.selector == "llm" else None
    history: List[List[Tuple[int, bool]]] = [[] for _ in range(cfg.n_shippers)]

    rows: List[Dict] = []
    cost_log: List[Dict] = []
    prev_edges: set = set()

    tracer = None
    if cfg.trace_path:
        import json as _json
        Path(cfg.trace_path).parent.mkdir(parents=True, exist_ok=True)
        tracer = open(cfg.trace_path, "w", encoding="utf-8")

        def _trace(obj: Dict) -> None:
            tracer.write(_json.dumps(obj, ensure_ascii=False, default=float)
                         + "\n")
    else:
        def _trace(obj: Dict) -> None:
            pass

    _trace({"type": "setup",
            "config": {"name": cfg.name, "selector": cfg.selector,
                       "n_shippers": cfg.n_shippers,
                       "n_carriers": cfg.n_carriers,
                       "n_rounds": cfg.n_rounds,
                       "candidate_pool_size": cfg.candidate_pool_size,
                       "fitness_regime": cfg.fitness_regime,
                       "shuffle": cfg.shuffle_candidates,
                       "dynamic_profile": cfg.dynamic_profile,
                       "show_capacity": cfg.show_capacity,
                       "rating_mode": cfg.rating_mode,
                       "llm_specs": list(cfg.llm_specs),
                       "seed": cfg.seed},
            "carriers": [
                {"carrier": j, "init_price": c.price,
                 "reliability": c.reliability, "capacity": c.capacity,
                 "specialty": c.specialty,
                 "fitness": c.base_profile_strength,
                 "rating0": c.rating, "rating_n0": c.rating_n,
                 "slots": economy.states[j].slots}
                for j, c in enumerate(carriers)]})

    for t in range(1, cfg.n_rounds + 1):
        shipper_order = rng.permutation(cfg.n_shippers)
        deg_snap = graph.carrier_degree().copy()
        economy.begin_round()
        active = economy.active_indices()

        def _process(i: int, pool, request: str, distance: int, tons: int,
                     ranked: List[int], meta: Dict) -> None:
            """Serve and record (always sequential, in shipper_order)."""
            if cfg.selector == "llm":
                cost_log.append({
                    "round": t, "shipper": i,
                    "backend": meta.get("backend", ""),
                    "model": meta.get("model", ""),
                    "input_tokens": meta.get("input_tokens", 0),
                    "output_tokens": meta.get("output_tokens", 0),
                    "cached": meta.get("cached", False),
                })
            pref_graph.add_match(i, int(ranked[0]))
            served_j, _depth = economy.serve_waterfall(
                [int(j) for j in ranked], distance=distance, tons=tons)
            if served_j is not None:
                graph.add_match(i, int(served_j))
                history[i].append((int(served_j), True))
            else:
                history[i].append((int(ranked[0]), False))
            rec = {"type": "decision", "round": t, "shipper": i,
                   "request": request, "distance": int(distance),
                   "tons": int(tons),
                   "pool": [int(j) for j, _ in pool],
                   "ranked": [int(x) for x in ranked],
                   "served": int(served_j) if served_j is not None else None,
                   "depth": int(_depth)}
            if cfg.selector == "llm":
                rec.update({
                    "backend": meta.get("backend", ""),
                    "model": meta.get("model", ""),
                    "fallback": "error" in meta,
                    "cached": bool(meta.get("cached", False)),
                    "choices_as_returned": meta.get("choices_as_returned"),
                    "reason": meta.get("reason", ""),
                    "prompt": meta.get("prompt", ""),
                    "raw_response": meta.get("raw_response", ""),
                })
            _trace(rec)

        # Parallel LLM calls: prompts depend only on the start-of-round
        # state, so "parallel calls, then serve sequentially in the original
        # random order" leaves the semantics unchanged. Random numbers
        # (candidate draws, request generation) are consumed on the main
        # thread in the usual order, preserving reproducibility. Only
        # show_capacity (within-round disclosure) is order-dependent and
        # therefore sequential.
        parallel = (cfg.selector == "llm" and not cfg.show_capacity
                    and cfg.llm_concurrency > 1)

        if not parallel:
            for i in shipper_order:
                pool = _build_candidate_pool(rng, carriers,
                                             cfg.candidate_pool_size,
                                             cfg.shuffle_candidates,
                                             active_indices=active)
                request, distance, tons = synthesize_request(rng)
                if cfg.selector == "llm":
                    summary = _summarize_history(history[int(i)],
                                                 cfg.history_summary_length)
                    cap_slots = [economy.states[j].slots for j, _ in pool]
                    cap_left = None
                    if cfg.show_capacity:
                        cap_left = [
                            max(0, economy.states[j].slots
                                - economy.states[j].used_this_round)
                            if economy.states[j].active else 0
                            for j, _ in pool
                        ]
                    ranked, meta = shippers[int(i)].select(
                        pool, deg_snap, request, summary,
                        distance=distance, tons=tons,
                        capacity_slots=cap_slots,
                        capacity_left=cap_left)
                elif cfg.selector == "random":
                    ranked, meta = select_random(pool, rng), {}
                elif cfg.selector == "pa":
                    ranked, meta = select_pure_pa(pool, deg_snap, rng=rng), {}
                elif cfg.selector == "conflicting":
                    ranked, meta = select_conflicting_attachment(
                        pool, deg_snap, rng=rng), {}
                elif cfg.selector == "greedy":
                    ranked, meta = select_score_greedy(pool), {}
                else:
                    raise ValueError(f"unknown selector {cfg.selector}")
                _process(int(i), pool, request, distance, tons, ranked, meta)
        else:
            inputs = []
            for i in shipper_order:
                pool = _build_candidate_pool(rng, carriers,
                                             cfg.candidate_pool_size,
                                             cfg.shuffle_candidates,
                                             active_indices=active)
                request, distance, tons = synthesize_request(rng)
                inputs.append((int(i), pool, request, distance, tons))

            def _decide(inp):
                i, pool, request, distance, tons = inp
                summary = _summarize_history(history[i],
                                             cfg.history_summary_length)
                cap_slots = [economy.states[j].slots for j, _ in pool]
                return shippers[i].select(pool, deg_snap, request, summary,
                                          distance=distance, tons=tons,
                                          capacity_slots=cap_slots)

            with ThreadPoolExecutor(max_workers=cfg.llm_concurrency) as ex:
                results = list(ex.map(_decide, inputs))
            for (i, pool, request, distance, tons), (ranked, meta) in zip(
                    inputs, results):
                _process(i, pool, request, distance, tons, ranked, meta)

        economy.end_round(t)

        metrics = snapshot(graph, prev_edges)
        metrics.update(economy.round_metrics())
        metrics["kappa_pref_top3"] = condensation_ratio(
            pref_graph.carrier_degree(), top_k=3)
        metrics.update({"round": t, "name": cfg.name,
                        "selector": cfg.selector, "seed": cfg.seed,
                        "fitness_regime": cfg.fitness_regime,
                        "llm_specs": ",".join(cfg.llm_specs),
                        "llm_mix_mode": cfg.llm_mix_mode,
                        "shuffle": cfg.shuffle_candidates,
                        "dynamic_profile": cfg.dynamic_profile,
                        "econ_enabled": cfg.econ_enabled,
                        "show_capacity": cfg.show_capacity,
                        "rating_mode": cfg.rating_mode})
        _trace({"type": "round_state", "round": t,
                "carriers": [
                    {"carrier": j, "active": s.active, "slots": s.slots,
                     "used": s.used_this_round, "price": s.price,
                     "unit_cost": s.unit_cost,
                     "served_total": s.served_total,
                     "rejected_total": s.rejected_total,
                     "cum_profit": s.cum_profit,
                     "cum_revenue": s.cum_revenue,
                     "rating": carriers[j].rating,
                     "rating_n": carriers[j].rating_n}
                    for j, s in enumerate(economy.states)],
                "degrees": graph.carrier_degree().tolist(),
                "pref_degrees": pref_graph.carrier_degree().tolist(),
                "metrics": {k: v for k, v in metrics.items()
                            if isinstance(v, (int, float, bool))}})

        rows.append(metrics)
        prev_edges = graph.edge_set()
        graph.decay_and_prune()
        pref_graph.decay_and_prune()

    if tracer is not None:
        tracer.close()

    df = pd.DataFrame(rows)
    df.attrs["final_degrees"] = graph.carrier_degree().tolist()
    df.attrs["cost_log"] = cost_log
    df.attrs["carrier_econ"] = [
        {"carrier": j, "slots": s.slots, "final_price": s.price,
         "served": s.served_total, "rejected": s.rejected_total,
         "cum_profit": s.cum_profit, "cum_revenue": s.cum_revenue,
         "active": s.active,
         "final_rating": carriers[j].rating,
         "final_rating_n": carriers[j].rating_n}
        for j, s in enumerate(economy.states)
    ]
    return df
