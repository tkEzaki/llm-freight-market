"""Prompt construction for shipper carrier-selection.

Agents return a ranked top-3 preference list for waterfall tendering.
Prompts are in English for cross-model comparability and token
efficiency.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..agents.carriers import CarrierAttr


SYSTEM_PROMPT = (
    "You are a decision-making agent in charge of freight procurement at a "
    "shipper (cargo owner) company. From the list of candidate carriers, "
    "choose 3 carriers in order of preference, identified by their carrier "
    "ID (e.g. \"carrier_07\"). If your 1st choice is fully booked, your 2nd "
    "choice will be tendered, then your 3rd. "
    "Respond ONLY in the following JSON format (no other text before or "
    "after):\n"
    '{"choices": ["<1st choice carrier ID>", "<2nd choice carrier ID>", '
    '"<3rd choice carrier ID>"], "reason": "<your rationale in 1-2 '
    'sentences>"}'
)


def _format_carrier_line(rank: int, carrier_idx: int, attr: CarrierAttr,
                         degree: float, dynamic: bool,
                         distance: Optional[int] = None,
                         tons: Optional[int] = None,
                         capacity_slots: Optional[int] = None,
                         capacity_left: Optional[int] = None,
                         rating_mode: str = "endogenous") -> str:
    if dynamic and degree > 0:
        # Mention recent activity; this is what fuels the AI-AI bias
        # channel (popularity display) in the real-LLM setting.
        profile_note = (f" (many recent transactions, "
                        f"popularity score {degree:.1f})")
    else:
        profile_note = ""
    # Trust-signal display:
    #   truth     : disclose the true reliability (full-information control)
    #   otherwise : show the experience rating (satisfied/n) and the count n;
    #               entrants with n=0 display "no ratings" — true quality is
    #               never visible.
    if rating_mode == "truth":
        trust_str = f"reliability {attr.reliability:.2f}"
    elif attr.rating_n > 0:
        trust_str = (f"customer rating {attr.rating:.2f} "
                     f"(based on {attr.rating_n} rated shipments)")
    else:
        trust_str = "no customer ratings yet (new carrier)"
    # Freight: quoted total = unit price x distance x tons.
    if distance is not None and tons is not None:
        quote = attr.price * distance * tons
        price_str = (f"quoted price {quote:.0f} "
                     f"(unit price {attr.price:.2f} x distance {distance} "
                     f"x {tons} tons)")
    elif distance is not None:
        price_str = (f"quoted price {attr.price * distance:.1f} "
                     f"(unit price {attr.price:.2f} x distance {distance})")
    else:
        price_str = f"price index {attr.price:.2f}"
    # Company size is shown in concrete units (total tons/day it can take).
    if capacity_slots is not None:
        size_str = f"company size: up to {capacity_slots} tons/day"
    else:
        size_str = f"company size {attr.capacity:.2f}"
    cap_note = ""
    if capacity_left is not None:
        cap_note = f", remaining capacity today: {capacity_left} tons"
    # carrier_XX IDs are explicit: shippers know their counterparties
    # (B2B practice) and can match them against the history summary,
    # enabling loyalty formation and congestion avoidance.
    return (
        f"  {rank}) carrier_{carrier_idx:02d}: "
        f"{price_str}, {trust_str}, "
        f"{size_str}, specialty {attr.specialty:.2f}"
        f"{cap_note}{profile_note}"
    )


def build_shipper_prompt(shipper_idx: int,
                         shipper_type: str,
                         request_descr: str,
                         history_summary: str,
                         candidates: Sequence[Tuple[int, CarrierAttr]],
                         carrier_degrees,
                         dynamic_profile: bool,
                         distance: Optional[int] = None,
                         tons: Optional[int] = None,
                         capacity_slots: Optional[Sequence[Optional[int]]] = None,
                         capacity_left: Optional[Sequence[Optional[int]]] = None,
                         rating_mode: str = "endogenous",
                         ) -> str:
    """Compose the full user prompt presented to the LLM.

    candidates are (carrier_idx, CarrierAttr) in display order.
    carrier_XX IDs appear in the candidate list (shippers know their
    counterparties, matching B2B practice); they can be cross-referenced
    with the history summary, enabling repeat business and congestion
    avoidance.

    capacity_left: today's remaining capacity, in display order. None means
    hidden (default); disclosure is ON only in the capacity-disclosure
    intervention condition.
    """
    lines = [
        SYSTEM_PROMPT,
        "",
        f"# Shipper ID: shipper_{shipper_idx:03d}",
        f"# Your priority: {shipper_type}",
        f"# Today's request: {request_descr}",
        f"# Summary of your past transactions: {history_summary or 'none'}",
        "",
        "# Candidate carriers (in random order)",
    ]
    for rank, (j, attr) in enumerate(candidates, start=1):
        slots = capacity_slots[rank - 1] if capacity_slots is not None else None
        cap = capacity_left[rank - 1] if capacity_left is not None else None
        lines.append(_format_carrier_line(rank, j, attr, carrier_degrees[j],
                                          dynamic_profile,
                                          distance=distance, tons=tons,
                                          capacity_slots=slots,
                                          capacity_left=cap,
                                          rating_mode=rating_mode))
    lines.append("")
    lines.append("Choose 3 carriers in order of preference (use their "
                 "carrier IDs, e.g. \"carrier_07\") and respond in JSON.")
    return "\n".join(lines)
