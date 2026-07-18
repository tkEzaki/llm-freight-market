"""Economic layer: carrier capacity, dynamic pricing, exit, welfare metrics.

Endogenizes the economic consequences of demand concentration:

  * Capacity   — carrier j can serve at most slots_j tons per round.
                 Excess requests fail (no match; the shipper pays a delay
                 cost).
  * Pricing    — prices rise under excess-demand pressure and fall when
                 demand is slack. The dynamic price is written back to
                 CarrierAttr.price, so prompts, pseudo, and greedy all see
                 the same price.
  * Exit       — carriers whose cumulative losses exceed a threshold leave
                 the market (off by default).
  * Welfare    — service rate / shipper surplus / carrier revenue
                 distribution / utilization / HHI.

Concentration only acquires a cost through this layer: the more demand
piles onto a favorite, the higher its failure rate and price, and the more
shipper surplus is eroded. This makes the trade-off between short-run match
efficiency and market health observable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .agents.carriers import CarrierAttr
from .network.metrics import gini_coefficient


@dataclass
class EconomyConfig:
    enabled: bool = True
    # ---- capacity ------------------------------------------------------
    # Capacity is measured in tons. System-wide capacity per round =
    # capacity_scale x n_shippers x mean load tons (1.3x expected total
    # demand). It is allocated in proportion to the carrier's capacity
    # attribute, so concentrated demand creates local shortages. A request
    # consumes capacity equal to its tonnage (30 tons uses 6x as much as 5).
    capacity_scale: float = 1.3
    min_carrier_tons: int = 10        # even the smallest firm takes 10 t/day
    # ---- prices --------------------------------------------------------
    # The price driver is excess-demand pressure,
    #   pressure = (tons served + tons rejected) / slots,
    # with p <- p * (1 + rate * (pressure - 1)).
    # A pure utilization driver lets a full-but-unpressured carrier keep
    # compounding price increases regardless of how deep the excess demand
    # is; the pressure form stops raising prices when a carrier is full but
    # rejects nothing, and raises them faster when applications are a
    # multiple of capacity — an adjustment proportional to the depth of
    # excess demand.
    price_adjust_rate: float = 0.10
    target_pressure: float = 1.0      # applications == capacity -> no change
    price_cap: float = 1.5            # upper bound on the price index
    unit_cost_ratio: float = 0.6      # unit cost = initial price x ratio
    # ---- shipper welfare -----------------------------------------------
    shipper_value: float = 1.2        # value of a served load (price units)
    failure_cost: float = 0.3         # delay / re-procurement cost of failure
    # ---- exit (off by default; reserved as an intervention axis) --------
    exit_enabled: bool = False
    exit_loss_threshold: float = 3.0  # exit once cumulative losses exceed this
    exit_min_rounds: int = 5          # no exit before this round
    # ---- information structure: experience ratings ----------------------
    # rating_mode:
    #   "endogenous": the displayed score is the experience rating
    #                 (satisfied transactions / n). Each completed
    #                 transaction draws satisfaction ~ Bernoulli(true
    #                 reliability) and updates the score. Carriers that are
    #                 never chosen never accumulate n, so information-driven
    #                 rich-get-richer dynamics are endogenous. [default]
    #   "static"    : the initial rating (n0 draws) is never updated
    #                 (feedback-severed control).
    #   "truth"     : the true reliability is disclosed directly
    #                 (full-information control).
    rating_mode: str = "endogenous"
    # Initial track record n0 = max(0, round(fitness * scale) - offset).
    # Maps heavy-tail fitness onto the depth of the track record: incumbents
    # start with n0 ~ 190 reviewed shipments, the lowest-fitness carriers
    # with n0 = 0, i.e. unrated entrants.
    rating_n0_scale: int = 200
    rating_n0_offset: int = 10


@dataclass
class CarrierState:
    slots: int                 # capacity per round (tons/day)
    price: float               # current price index (dynamic, per ton-distance)
    unit_cost: float           # cost per ton-distance
    active: bool = True
    used_this_round: int = 0       # tons served this round
    rejected_this_round: int = 0   # tons rejected this round (price pressure)
    served_total: int = 0          # cumulative served count
    rejected_total: int = 0        # cumulative rejection count
    cum_profit: float = 0.0
    cum_revenue: float = 0.0
    rating_n: int = 0              # cumulative rated transactions
    rating_pos: int = 0            # of which satisfied


class Economy:
    """Round-by-round economic bookkeeping wired into the simulation loop."""

    def __init__(self, carriers: List[CarrierAttr], n_shippers: int,
                 cfg: EconomyConfig,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.carriers = carriers
        self.n_shippers = n_shippers
        # Dedicated rng for satisfaction draws (kept separate so the main
        # stream's random-number consumption is unchanged).
        self.rng = rng if rng is not None else np.random.default_rng(0)

        from .agents.shipper_llm import TONS_CHOICES
        cap_attrs = np.array([c.capacity for c in carriers], dtype=float)
        mean_tons = float(np.mean(TONS_CHOICES))
        total_tons = cfg.capacity_scale * n_shippers * mean_tons
        raw = total_tons * cap_attrs / cap_attrs.sum()
        self.states: List[CarrierState] = [
            CarrierState(
                slots=max(cfg.min_carrier_tons, int(round(raw[j]))),
                price=carriers[j].price,
                unit_cost=carriers[j].price * cfg.unit_cost_ratio,
            )
            for j in range(len(carriers))
        ]
        # ---- initialize experience ratings -----------------------------
        # n0 = depth of the track record (from heavy-tail fitness). The
        # initial score is n0 Bernoulli draws from the true reliability, so
        # unlucky underrated but capable carriers arise naturally. In truth
        # mode the true value is written back directly.
        for j, (c, s) in enumerate(zip(carriers, self.states)):
            if cfg.rating_mode == "truth":
                c.rating = c.reliability
                c.rating_n = -1          # marker for "truth disclosed"
                continue
            n0 = max(0, int(round(c.base_profile_strength
                                  * cfg.rating_n0_scale))
                     - cfg.rating_n0_offset)
            pos = int(self.rng.binomial(n0, c.reliability)) if n0 > 0 else 0
            s.rating_n, s.rating_pos = n0, pos
            self._write_back_rating(j)
        # Per-round shipper welfare and transaction bookkeeping.
        self._round_surplus: List[float] = []
        self._round_prices: List[float] = []
        self._round_served_by: Dict[int, int] = {}
        self._round_requests = 0
        self._round_tenders = 0        # tender attempts (incl. waterfall)
        self._round_rejections = 0     # tenders rejected for lack of capacity
        self._round_displaced = 0      # loads served at 2nd choice or lower

    # ------------------------------------------------------------------
    def _write_back_rating(self, j: int) -> None:
        """Write the observed rating back to CarrierAttr (n=0 -> prior 0.5)."""
        s = self.states[j]
        self.carriers[j].rating = (s.rating_pos / s.rating_n
                                   if s.rating_n > 0 else 0.5)
        self.carriers[j].rating_n = s.rating_n

    def active_indices(self) -> List[int]:
        return [j for j, s in enumerate(self.states) if s.active]

    def begin_round(self) -> None:
        for s in self.states:
            s.used_this_round = 0
            s.rejected_this_round = 0
        self._round_surplus = []
        self._round_prices = []
        self._round_served_by = {}
        self._round_requests = 0
        self._round_tenders = 0
        self._round_rejections = 0
        self._round_displaced = 0

    def serve_waterfall(self, ranked: List[int], distance: float = 1.0,
                        tons: int = 1) -> Tuple[Optional[int], int]:
        """Waterfall tendering: offer down the ranked list, first fit serves.

        Returns (serving carrier idx or None, tender depth reached).
        Mirrors routing-guide practice: if the first choice rejects the
        tender (no remaining capacity), the load goes to the second choice;
        if all three ranked carriers are full, the load fails.

        tons: load tonnage. Capacity use, rejection pressure, and money all
        scale with tons.
        distance: transport distance. Payment = unit price x distance x
        tons; monetary aggregates are on the tons x distance scale, while
        welfare (surplus) is normalized per ton-distance.
        """
        self._round_requests += 1
        if not self.cfg.enabled:
            j = ranked[0]
            self._record_served(j, self.states[j], distance, tons)
            return j, 1
        for depth, j in enumerate(ranked, start=1):
            s = self.states[j]
            self._round_tenders += 1
            if s.active and s.used_this_round + tons <= s.slots:
                self._record_served(j, s, distance, tons)
                if depth > 1:
                    self._round_displaced += 1
                return j, depth
            s.rejected_total += 1
            s.rejected_this_round += tons   # pressure is in tons
            self._round_rejections += 1
        self._round_surplus.append(-self.cfg.failure_cost)
        return None, len(ranked)

    def _record_served(self, carrier_idx: int, s: CarrierState,
                       distance: float = 1.0, tons: int = 1) -> None:
        s.used_this_round += tons
        s.served_total += 1
        # Endogenous rating update: satisfaction of a completed transaction
        # ~ Bernoulli(true quality). The write-back to the displayed
        # attribute happens in end_round, so prompts always see the
        # start-of-round value.
        if self.cfg.rating_mode == "endogenous":
            s.rating_n += 1
            if self.rng.random() < self.carriers[carrier_idx].reliability:
                s.rating_pos += 1
        # Money is on the tons x distance scale: payment = price x dist x tons.
        s.cum_profit += (s.price - s.unit_cost) * distance * tons
        s.cum_revenue += s.price * distance * tons
        # Welfare and price metrics are recorded per ton-distance (unit
        # price basis) so they are insensitive to the round's demand mix.
        self._round_surplus.append(self.cfg.shipper_value - s.price)
        self._round_prices.append(s.price)
        self._round_served_by[carrier_idx] = \
            self._round_served_by.get(carrier_idx, 0) + 1

    # ------------------------------------------------------------------
    def end_round(self, round_no: int) -> None:
        # Write back experience ratings (works even with econ disabled).
        # This is where the "prompts see start-of-round ratings" semantics
        # is guaranteed.
        if self.cfg.rating_mode == "endogenous":
            for j in range(len(self.states)):
                self._write_back_rating(j)
        if not self.cfg.enabled:
            return
        for j, s in enumerate(self.states):
            if not s.active:
                continue
            # Excess-demand pressure = (served + rejected) / slots.
            # 1.0 means applications exactly filled capacity.
            pressure = (s.used_this_round + s.rejected_this_round) / s.slots
            s.price *= 1.0 + self.cfg.price_adjust_rate * (
                pressure - self.cfg.target_pressure)
            s.price = float(np.clip(s.price, s.unit_cost * 1.05,
                                    self.cfg.price_cap))
            # Write the dynamic price back to the attribute, which prompts,
            # greedy, and pseudo all read.
            self.carriers[j].price = s.price
            if (self.cfg.exit_enabled
                    and round_no >= self.cfg.exit_min_rounds
                    and s.cum_profit < -self.cfg.exit_loss_threshold):
                s.active = False

    # ------------------------------------------------------------------
    def round_metrics(self) -> Dict[str, float]:
        """Economic observables for the round that just ended."""
        n_served = len(self._round_prices)
        n_req = max(self._round_requests, 1)
        served_counts = np.array(list(self._round_served_by.values()),
                                 dtype=float)
        shares = served_counts / served_counts.sum() if n_served else served_counts
        revenues = np.array([s.cum_revenue for s in self.states])
        top3_rev = float(np.sort(revenues)[-3:].sum() / revenues.sum()) \
            if revenues.sum() > 0 else 0.0
        utils = [s.used_this_round / s.slots
                 for s in self.states if s.active]
        pressures = [(s.used_this_round + s.rejected_this_round) / s.slots
                     for s in self.states if s.active]
        return {
            "mean_pressure": float(np.mean(pressures)) if pressures else 0.0,
            "service_rate": n_served / n_req,
            "displacement_rate": self._round_displaced / max(n_served, 1),
            "tender_rejection_rate": self._round_rejections
                / max(self._round_tenders, 1),
            "n_unserved": self._round_requests - n_served,
            "mean_price": float(np.mean(self._round_prices)) if n_served else 0.0,
            "price_std": float(np.std(self._round_prices)) if n_served else 0.0,
            "shipper_surplus": float(np.mean(self._round_surplus))
                if self._round_surplus else 0.0,
            "hhi_round": float(np.sum(shares ** 2)) if n_served else 0.0,
            "mean_util": float(np.mean(utils)) if utils else 0.0,
            "revenue_gini": gini_coefficient(revenues),
            "revenue_top3_share": top3_rev,
            "n_active_carriers": float(sum(s.active for s in self.states)),
            "n_loss_carriers": float(sum(s.cum_profit < 0 for s in self.states
                                         if s.active)),
        }
