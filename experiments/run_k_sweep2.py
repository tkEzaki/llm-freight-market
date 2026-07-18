# -*- coding: utf-8 -*-
"""All experimental phases reported in the paper (resumable driver).

Information structure:
  * No "profile quality" attribute; the true reliability is never shown.
    Shippers see the experience rating only, e.g.
    "customer rating 0.83 (based on 214 rated shipments)".
  * The initial track record n0 follows a heavy-tailed distribution;
    carriers with n0 = 0 display "no customer ratings yet".
  * Ratings update endogenously: each completed transaction draws
    satisfaction ~ Bernoulli(true quality). Carriers that are never chosen
    never gain rating precision (information-driven rich-get-richer).

Phases (all: heavy_tail, 50 shippers x 20 carriers x 30 days, full traces):
  v1  : 4 reference rules x L in {3,5,10,20}
  v2  : GPT (endogenous rating) x L in {3,5,10,20}
  v2b : L=15 for reference rules and GPT
  v3  : information ablation (GPT): static / truth at L in {5,20}
  v4  : Claude / Gemini x L in {5,20}
  v4b : Claude / Gemini x L in {3,10,15}
  v5  : interventions (GPT): poly3 / capacity disclosure / fixed order /
        popularity display
  vclaude, vgemini, vgptint, vpoly : vendor-specific subsets of the above
        (useful when one vendor's daily quota is exhausted; tags are shared,
        so completed cells are skipped)
  va1 : price-adjustment-rate counterfactual (0.05 / 0.02, seeds 0-2)
  va2 : static, seeds 5-9   (n=10 for the dispersion comparison)
  va3 : L in {12,13} for GPT and reference rules
  va5 : endogenous, seeds 5-9 (n=10)
  va6 : truth, seeds 5-9      (n=10)

Usage:  python -X utf8 experiments/run_k_sweep2.py [v1 v2 ...]
Resumable: cells with an existing rounds CSV are skipped. Aborts if the
fallback rate exceeds 5%. Output: results/k_sweep2/
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.base import LLMQuotaError          # noqa: E402
from src.runner import CellConfig, run_cell     # noqa: E402

OUT = ROOT / "results" / "k_sweep2"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "cache"

FULL = dict(n_shippers=50, n_carriers=20, n_rounds=30,
            fitness_regime="heavy_tail")
GPT = ("openai:gpt-5.4-mini",)
CLAUDE = ("anthropic:claude-haiku-4-5",)
GEMINI = ("google:gemini-3.5-flash",)
POLY3 = GPT + CLAUDE + GEMINI

FALLBACK_LIMIT = 0.05
SEEDS = [0, 1, 2, 3, 4]


def run_one(tag: str, seed: int, k: int, **kw) -> None:
    csv_path = OUT / f"rounds_{tag}.csv"
    if csv_path.exists():
        print(f"{tag:26s} SKIP (already done)", flush=True)
        return
    t0 = time.time()
    cfg = CellConfig(name=f"k2-{tag}", seed=seed, candidate_pool_size=k,
                     trace_path=str(OUT / f"trace_{tag}.jsonl"),
                     **FULL, **kw)
    df = run_cell(cfg)
    df.to_csv(csv_path, index=False)
    cl = df.attrs.get("cost_log", [])
    fb = sum(1 for c in cl if not c.get("backend")) if cl else 0
    fb_rate = fb / len(cl) if cl else 0.0
    print(f"{tag:26s} kappa_pref={df['kappa_pref_top3'].iloc[-1]:.3f} "
          f"kappa={df['kappa_top3'].iloc[-1]:.3f} "
          f"service={df['service_rate'].iloc[-1]:.3f} "
          f"price={df['mean_price'].iloc[-1]:.3f} "
          f"fallback={fb}/{len(cl)} wall={time.time() - t0:.0f}s", flush=True)
    if cl and fb_rate > FALLBACK_LIMIT:
        print(f"ABORT: fallback rate {fb_rate:.1%} > {FALLBACK_LIMIT:.0%} "
              f"in {tag}", flush=True)
        sys.exit(2)


def cache_kw(arm: str, k: int, seed: int) -> dict:
    return dict(cache_path=str(CACHE / f"k2_{arm}_k{k}_s{seed}.jsonl"))


# ---------------------------------------------------------------- phases
def v1() -> None:
    for seed in SEEDS:
        for sel in ["random", "pa", "conflicting", "greedy"]:
            for k in [3, 5, 10, 20]:
                run_one(f"{sel}-k{k}-s{seed}", seed, k, selector=sel)


def v2() -> None:
    for seed in SEEDS:
        for k in [3, 5, 10, 20]:
            run_one(f"gpt-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=GPT, **cache_kw("gpt", k, seed))


def v2b() -> None:
    """Add L=15 (localizes the threshold between flat L10 and the L20 jump)."""
    for seed in SEEDS:
        for sel in ["random", "pa", "conflicting", "greedy"]:
            run_one(f"{sel}-k15-s{seed}", seed, 15, selector=sel)
        run_one(f"gpt-k15-s{seed}", seed, 15, selector="llm",
                llm_specs=GPT, **cache_kw("gpt", 15, seed))


def v3() -> None:
    """Information ablation: endogenous (v2) / static / truth."""
    for seed in SEEDS:
        for k in [5, 20]:
            run_one(f"gpt-static-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=GPT, rating_mode="static",
                    **cache_kw("gpt_static", k, seed))
            run_one(f"gpt-truth-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=GPT, rating_mode="truth",
                    **cache_kw("gpt_truth", k, seed))


def v4() -> None:
    for seed in SEEDS:
        for k in [5, 20]:
            run_one(f"claude-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=CLAUDE, **cache_kw("claude", k, seed))
            run_one(f"gemini-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=GEMINI, **cache_kw("gemini", k, seed))


def v4b() -> None:
    """Extend v4 to L in {3,10,15} (L5/L20 are covered by v4)."""
    for seed in SEEDS:
        for k in [3, 10, 15]:
            run_one(f"claude-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=CLAUDE, **cache_kw("claude", k, seed))
            run_one(f"gemini-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=GEMINI, **cache_kw("gemini", k, seed))


def v5() -> None:
    for seed in SEEDS:
        for k in [5, 20]:
            run_one(f"poly3-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=POLY3, llm_mix_mode="polyculture",
                    **cache_kw("poly3", k, seed))
        run_one(f"showcap-k20-s{seed}", seed, 20, selector="llm",
                llm_specs=GPT, show_capacity=True,
                **cache_kw("showcap", 20, seed))
        run_one(f"noshuffle-k20-s{seed}", seed, 20, selector="llm",
                llm_specs=GPT, shuffle_candidates=False,
                **cache_kw("noshuffle", 20, seed))
        run_one(f"popularity-k20-s{seed}", seed, 20, selector="llm",
                llm_specs=GPT, dynamic_profile=True,
                **cache_kw("popularity", 20, seed))


def vclaude() -> None:
    """Claude cells only (tags shared with v4/v4b; completed cells skip)."""
    for seed in SEEDS:
        for k in [3, 5, 10, 15, 20]:
            run_one(f"claude-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=CLAUDE, **cache_kw("claude", k, seed))


def vgemini() -> None:
    """Gemini cells only (tags shared with v4/v4b; completed cells skip).

    Ordered by importance for the threshold analysis (L20/L15 first) so
    that a daily-quota interruption costs the least; on quota exhaustion
    the run fails fast and the same command resumes the remainder later.
    """
    for k in [20, 15, 10, 5, 3]:
        for seed in SEEDS:
            run_one(f"gemini-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=GEMINI, **cache_kw("gemini", k, seed))


def vgptint() -> None:
    """GPT-only interventions (tags shared with v5; poly3 excluded)."""
    for seed in SEEDS:
        run_one(f"showcap-k20-s{seed}", seed, 20, selector="llm",
                llm_specs=GPT, show_capacity=True,
                **cache_kw("showcap", 20, seed))
        run_one(f"noshuffle-k20-s{seed}", seed, 20, selector="llm",
                llm_specs=GPT, shuffle_candidates=False,
                **cache_kw("noshuffle", 20, seed))
        run_one(f"popularity-k20-s{seed}", seed, 20, selector="llm",
                llm_specs=GPT, dynamic_profile=True,
                **cache_kw("popularity", 20, seed))


def vpoly() -> None:
    """Vendor-mix cells only (tags shared with v5)."""
    for seed in SEEDS:
        for k in [5, 20]:
            run_one(f"poly3-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=POLY3, llm_mix_mode="polyculture",
                    **cache_kw("poly3", k, seed))


def va1() -> None:
    """Price-sensitivity counterfactual (Fig. 2f): slow the price response.

    GPT L=20 endogenous x price_adjust_rate in {0.05, 0.02} x seeds 0-2.
    The base rate 0.10 reuses the existing gpt-k20-s* cells.
    """
    for rate, tag in [(0.05, "pr05"), (0.02, "pr02")]:
        for seed in [0, 1, 2]:
            run_one(f"gpt-{tag}-k20-s{seed}", seed, 20, selector="llm",
                    llm_specs=GPT, price_adjust_rate=rate,
                    **cache_kw(f"gpt_{tag}", 20, seed))


def va2() -> None:
    """Static condition, seeds 5-9 (n=10 for the dispersion comparison)."""
    for seed in [5, 6, 7, 8, 9]:
        run_one(f"gpt-static-k20-s{seed}", seed, 20, selector="llm",
                llm_specs=GPT, rating_mode="static",
                **cache_kw("gpt_static", 20, seed))


def va3() -> None:
    """Threshold resolution: GPT L in {12,13} x 5 seeds (+ reference rules)."""
    for seed in SEEDS:
        for k in [12, 13]:
            for sel in ["random", "pa", "conflicting", "greedy"]:
                run_one(f"{sel}-k{k}-s{seed}", seed, k, selector=sel)
            run_one(f"gpt-k{k}-s{seed}", seed, k, selector="llm",
                    llm_specs=GPT, **cache_kw("gpt", k, seed))


def va5() -> None:
    """Endogenous condition, seeds 5-9 (n=10)."""
    for seed in [5, 6, 7, 8, 9]:
        run_one(f"gpt-k20-s{seed}", seed, 20, selector="llm",
                llm_specs=GPT, **cache_kw("gpt", 20, seed))


def va6() -> None:
    """Truth condition, seeds 5-9 (all three information conditions at n=10)."""
    for seed in [5, 6, 7, 8, 9]:
        run_one(f"gpt-truth-k20-s{seed}", seed, 20, selector="llm",
                llm_specs=GPT, rating_mode="truth",
                **cache_kw("gpt_truth", 20, seed))


PHASES = {"v1": v1, "v2": v2, "v2b": v2b, "v3": v3, "v4": v4,
          "v4b": v4b, "v5": v5,
          "vclaude": vclaude, "vgemini": vgemini,
          "vgptint": vgptint, "vpoly": vpoly,
          "va1": va1, "va2": va2, "va3": va3, "va5": va5, "va6": va6}


def main() -> None:
    names = [a.lower() for a in sys.argv[1:]] or list(PHASES)
    unknown = [n for n in names if n not in PHASES]
    if unknown:
        sys.exit(f"unknown phase(s): {unknown} — choose from {list(PHASES)}")
    t0 = time.time()
    for n in names:
        print(f"===== PHASE {n.upper()} =====", flush=True)
        try:
            PHASES[n]()
        except LLMQuotaError as e:
            print(f"QUOTA EXHAUSTED: {e}\n"
                  f"Re-run the same command to resume after topping up.",
                  flush=True)
            sys.exit(3)
    print(f"ALL DONE total_wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
