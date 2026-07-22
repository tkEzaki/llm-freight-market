# -*- coding: utf-8 -*-
"""Paper figures (Nature-style, 180 mm wide) + numeric digest for Results.

Writes fig1_panels, fig2..fig6, figA1, figA2 (.pdf + .png) and
stats_digest.txt, which lists every number cited in the Results text,
to figs/.

Reads a fresh run from results/k_sweep2/ if present, otherwise the
archived runs shipped in data/k_sweep2/ (gzipped traces are read
transparently), so the figures reproduce without any API calls.

Run: python -X utf8 experiments/make_paper_figs.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Fresh runs write to results/; the repository ships the reported runs in
# data/ (rounds as plain CSV, traces gzipped). results/ takes precedence.
_RESULTS = ROOT / "results" / "k_sweep2"
_SHIPPED = ROOT / "data" / "k_sweep2"
DATA = _RESULTS if _RESULTS.exists() else _SHIPPED
OUT = ROOT / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- Nature-ish style ----------------------------------
MM = 1 / 25.4
W = 180 * MM
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

C_GPT = "#2a78d6"
C_CLAUDE = "#eda100"
C_GEMINI = "#1baf7a"
C_GREEDY = "#008300"
C_GRAY = "#767471"
C_ACC = "#c0502d"
C_PURPLE = "#4a3aa7"
K_RAMP = {3: "#a8c4ec", 5: "#7fa9e2", 10: "#5089d3", 15: "#2a6bbf", 20: "#123f80"}
KS = [3, 5, 10, 15, 20]
SEEDS = range(5)
DIGEST = []


def note(s):
    DIGEST.append(s)
    print(s)


def rounds(tag):
    p = DATA / f"rounds_{tag}.csv"
    return pd.read_csv(p) if p.exists() else None


def tail5(tag, col):
    df = rounds(tag)
    return float(df[col].tail(5).mean()) if df is not None else None


def seed_vals(arm, k, col):
    vs = [tail5(f"{arm}-k{k}-s{s}", col) for s in SEEDS]
    return [v for v in vs if v is not None]


def panel(ax, letter, dx=-0.18, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved:", OUT / f"{name}.png")


def trace_iter(tag):
    plain = DATA / f"trace_{tag}.jsonl"
    if plain.exists():
        f = open(plain, encoding="utf-8")
    else:
        import gzip
        f = gzip.open(plain.with_suffix(".jsonl.gz"), "rt", encoding="utf-8")
    with f:
        for line in f:
            yield json.loads(line)


# ====================================================================
# Fig 1 panels b–d  (a = TikZ, assembled in LaTeX)
# ====================================================================
def fig1():
    fig = plt.figure(figsize=(W, 52 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.06, right=0.98,
                          top=0.88, bottom=0.17)

    # (b) carrier population
    ax = fig.add_subplot(gs[0])
    panel(ax, "b")
    setup = next(trace_iter("trace_gpt-k20-s0".replace("trace_", "")))
    car = setup["carriers"]
    xs = [c["init_price"] for c in car]
    ys = [c["reliability"] for c in car]
    ss = [c["slots"] for c in car]
    n0 = [c["rating_n0"] for c in car]
    from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
    cmap = LinearSegmentedColormap.from_list("b", ["#dce9f8", "#123f80"])
    norm = SymLogNorm(linthresh=1, vmin=0, vmax=max(n0))
    sc = ax.scatter(xs, ys, s=[v * 1.8 for v in ss],
                    c=n0, cmap=cmap, norm=norm,
                    edgecolors="white", linewidths=0.5, zorder=3)
    for c, x, y in zip(car, xs, ys):
        if c["rating_n0"] == 0:
            ax.scatter([x], [y], s=c["slots"] * 1.8, facecolors="none",
                       edgecolors="k", linewidths=0.8, zorder=4,
                       linestyle="--")
    i18 = next(c for c in car if c["carrier"] == 18)
    ax.annotate("Carrier 18\n(cheap, reliable)",
                (i18["init_price"], i18["reliability"]),
                xytext=(0.15, 0.40), textcoords=ax.transAxes,
                fontsize=6, color=C_ACC, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_ACC,
                                relpos=(0, 1)))
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02,
                      ticks=[0, 1, 10, 100])
    cb.set_label("Initial track record $n_0$ (log)", fontsize=6)
    cb.ax.set_yticklabels(["0", "1", "10", "100"])
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    ax.set_xlabel("Initial unit price")
    ax.set_ylabel("True reliability (hidden)")
    ax.set_title("Carrier population (bubble = capacity)", fontsize=7)

    # (c) reference dynamics: greedy vs random
    ax = fig.add_subplot(gs[1])
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.025, pos.y0, pos.width, pos.height])
    panel(ax, "c")
    for arm, color, label in [("greedy", C_GREEDY, "Deterministic scoring"),
                              ("random", C_GRAY, "Random choice")]:
        ys = np.array([rounds(f"{arm}-k20-s{s}")["kappa_pref_top3"].values
                       for s in SEEDS])
        ax.fill_between(range(1, 31), ys.min(0), ys.max(0), color=color,
                        alpha=0.18, lw=0)
        ax.plot(range(1, 31), ys.mean(0), color=color, lw=1.2, label=label)
    ax.set_xlabel("Day")
    ax.set_ylabel(r"Concentration $\kappa$")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Reference decision rules ($L=20$)", fontsize=7)

    # (d) design summary
    ax = fig.add_subplot(gs[2])
    panel(ax, "d")
    ax.axis("off")
    rowsx = [
        ("Market", "50 shippers × 20 carriers\n× 30 days"),
        ("Agents", "GPT / Claude / Gemini"),
        ("Lever 1", "exposure $L \\in \\{3,5,10,15,20\\}$"),
        ("Lever 2", "trust signal:\ntruth / static / endogenous"),
        ("Lever 3", "capacity disclosure, list order,\n"
                    "popularity display, vendor mix"),
        ("Audit", "every prompt, response, and\nmarket state logged"),
    ]
    y = 1.00
    for k, v in rowsx:
        ax.text(0.0, y, k, fontsize=6, fontweight="bold", va="top")
        ax.text(0.26, y, v, fontsize=6, va="top")
        y -= 0.135 if "\n" not in v else 0.20
    ax.set_title("Experimental design", fontsize=7, loc="left")
    save(fig, "fig1_panels")


# ====================================================================
# Fig 2 — consensus on day one; prices partially heal it
# ====================================================================
def fig2():
    fig = plt.figure(figsize=(W, 95 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.50, hspace=0.55, left=0.07,
                          right=0.955, top=0.93, bottom=0.09)

    # (a) day-1 votes: bars = mean across runs (rank-sorted), dots = runs
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    per_run = []
    for s in SEEDS:
        votes = Counter()
        for r in trace_iter(f"gpt-k20-s{s}"):
            if r.get("type") == "decision" and r["round"] == 1:
                votes[r["ranked"][0]] += 1
        per_run.append(sorted(votes.values(), reverse=True) + [0] * 20)
    ranked = np.array([row[:20] for row in per_run], dtype=float)
    means = ranked.mean(0)
    ax.bar(range(20), means,
           color=[C_ACC if i == 0 else C_GPT for i in range(20)], width=0.7,
           alpha=0.85)
    for s in range(len(per_run)):
        ax.scatter(range(20), ranked[s], color="k", s=5, zorder=3,
                   edgecolors="white", linewidths=0.3, alpha=0.8)
    note(f"[fig2a] day1 top-rank votes per run: {ranked[:,0].tolist()}")
    ax.set_xticks([])
    ax.set_xlabel("Carriers (rank-sorted per run)")
    ax.set_ylabel("Day-1 first choices")
    ax.set_title("Day 1: choices pile onto a few carriers", fontsize=7)
    # cross-condition first-day tally
    d1 = []
    for tag in ["gpt-k20-s0", "gpt-truth-k20-s0", "gpt-static-k20-s0",
                "claude-k20-s0", "gemini-k20-s0"]:
        v = Counter()
        for r in trace_iter(tag):
            if r.get("type") == "decision" and r["round"] == 1:
                v[r["ranked"][0]] += 1
        d1.append(v.most_common(1)[0][1])
    ax.text(0.30, 0.78, f"Top-1 votes across models &\ninformation structures:"
            f" {min(d1)}–{max(d1)} / 50\n(dots: individual runs)",
            transform=ax.transAxes, fontsize=6, va="top")
    note(f"[fig2a] day1 top1 votes across 5 conditions: {d1} (range {min(d1)}-{max(d1)})")

    # (b) kappa time series, 5 seeds
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    ys = np.array([rounds(f"gpt-k20-s{s}")["kappa_pref_top3"].values
                   for s in SEEDS])
    for row in ys:
        ax.plot(range(1, 31), row, color=C_GPT, lw=0.5, alpha=0.45)
    ax.plot(range(1, 31), ys.mean(0), color=C_GPT, lw=1.6, label="GPT, $L=20$")
    rn = np.array([rounds(f"random-k20-s{s}")["kappa_pref_top3"].values
                   for s in SEEDS])
    ax.fill_between(range(1, 31), rn.min(0), rn.max(0), color=C_GRAY,
                    alpha=0.25, lw=0, label="Random reference")
    ax.set_xlabel("Day")
    ax.set_ylabel(r"Concentration $\kappa$")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.set_title("Consensus erodes, then persists", fontsize=7)
    note(f"[fig2b] gpt k20 day1 mean kappa={ys[:,0].mean():.2f}, "
         f"final mean={ys[:,-1].mean():.2f} range [{ys[:,-1].min():.2f},{ys[:,-1].max():.2f}]")

    # (c) hub share and price (seed 0)
    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c")
    share, price = [], []
    day_votes = {}
    for r in trace_iter("gpt-k20-s0"):
        if r.get("type") == "decision":
            day_votes.setdefault(r["round"], []).append(r["ranked"][0])
        elif r.get("type") == "round_state":
            hub = next(c for c in r["carriers"] if c["carrier"] == 18)
            price.append(hub["price"])
    for t in sorted(day_votes):
        share.append(sum(1 for j in day_votes[t] if j == 18) / 50)
    ax.plot(range(1, 31), share, color=C_ACC, lw=1.3)
    ax2 = ax.twinx()
    ax2.plot(range(1, 31), price, color=C_ACC, lw=1.0, ls="--")
    ax2.set_ylabel("Unit price (dashed)", fontsize=6)
    ax2.tick_params(labelsize=5.5)
    ax2.spines["top"].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=C_ACC, lw=1.3, label="First-choice share"),
        Line2D([], [], color=C_ACC, lw=1.0, ls="--", label="Unit price")],
        frameon=False, fontsize=6, loc="center right")
    ax.set_xlabel("Day")
    ax.set_ylabel("First-choice share")
    ax.set_ylim(0, 0.8)
    ax.set_title("The day-1 favorite prices itself out", fontsize=7)
    note(f"[fig2c] hub share day1={share[0]:.2f} -> day30={share[-1]:.2f}; "
         f"price day1={price[0]:.2f} -> max={max(price):.2f} (cap 1.5) reached day "
         f"{int(np.argmax(np.array(price) >= 1.49)) + 1}")

    # (d) rating repair (seed 0)
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "d")
    setup, ratings = None, {}
    for r in trace_iter("gpt-k20-s0"):
        if r.get("type") == "setup":
            setup = r
        elif r.get("type") == "round_state":
            for c in r["carriers"]:
                ratings.setdefault(c["carrier"], []).append(c["rating"])
    true_rel = {c["carrier"]: c["reliability"] for c in setup["carriers"]}
    focus = [14, 13, 3, 11, 2]
    cols = [C_GPT, C_GEMINI, C_CLAUDE, C_PURPLE, C_GREEDY]
    for j, col in zip(focus, cols):
        ax.plot(range(1, 31), ratings[j], color=col, lw=1.0)
        ax.axhline(true_rel[j], color=col, lw=0.6, ls=":", alpha=0.7)
    ax.set_xlabel("Day")
    ax.set_ylabel("Displayed rating")
    ax.set_ylim(0.2, 1.0)
    ax.set_title("Unrated entrants earn accurate ratings", fontsize=7)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=C_GRAY, lw=0.8, ls=":",
                              label="True reliability")],
              frameon=False, fontsize=6, loc="lower right",
              bbox_to_anchor=(1.04, 0.0), borderaxespad=0.0)

    # (e) welfare (5-seed mean ± range)
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "e")
    for col_name, color, label in [("mean_price", C_ACC, "Mean paid price"),
                                   ("shipper_surplus", C_GPT, "Shipper surplus")]:
        ys = np.array([rounds(f"gpt-k20-s{s}")[col_name].values for s in SEEDS])
        ax.fill_between(range(1, 31), ys.min(0), ys.max(0), color=color,
                        alpha=0.18, lw=0)
        ax.plot(range(1, 31), ys.mean(0), color=color, lw=1.2, label=label)
    ax.set_xlabel("Day")
    ax.set_ylabel("Index")
    ax.legend(frameon=False)
    ax.set_title("Shippers pay for the correction", fontsize=7)

    # (f) price-stickiness counterfactual: day-1 hub share trajectory
    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "f")

    def fav_share_series(tag):
        day_votes = {}
        for r in trace_iter(tag):
            if r.get("type") == "decision":
                day_votes.setdefault(r["round"], []).append(r["ranked"][0])
        hub = Counter(day_votes[1]).most_common(1)[0][0]
        return [sum(1 for j in day_votes[t] if j == hub) / 50
                for t in sorted(day_votes)]

    for arm, color, label in [("gpt", C_GPT, "0.10 (base)"),
                              ("gpt-pr05", "#7fa9e2", "0.05"),
                              ("gpt-pr02", C_ACC, "0.02 (sticky)")]:
        ys = np.array([fav_share_series(f"{arm}-k20-s{s}") for s in range(3)])
        ax.plot(range(1, 31), ys.mean(0), color=color, lw=1.2, label=label)
        note(f"[fig2f] rate {label}: hub share d2={ys[:,1].mean():.2f} "
             f"d30={ys[:,-1].mean():.2f}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Day-1 favorite's share")
    ax.legend(frameon=False, fontsize=6, title="Price adj. rate",
              title_fontsize=6)
    ax.set_title("Prices set the pace, capacity the endpoint", fontsize=7)
    save(fig, "fig2")


# ====================================================================
# Fig 3 — exposure threshold
# ====================================================================
def fig3():
    fig = plt.figure(figsize=(W, 94 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.45, hspace=0.60, left=0.07,
                          right=0.98, top=0.87, bottom=0.09)

    def dose(ax, col, ylabel, title, letter):
        panel(ax, letter)
        for arm, color, label in [("gpt", C_GPT, "GPT"),
                                  ("claude", C_CLAUDE, "Claude"),
                                  ("gemini", C_GEMINI, "Gemini"),
                                  ("greedy", C_GREEDY, "Deterministic"),
                                  ("random", C_GRAY, "Random")]:
            ks_arm = KS
            means, los, his, ns = [], [], [], []
            for k in ks_arm:
                vs = seed_vals(arm, k, col)
                means.append(np.mean(vs) if vs else np.nan)
                los.append(min(vs) if vs else np.nan)
                his.append(max(vs) if vs else np.nan)
                ns.append(len(vs))
            ls = "--" if arm in ("random",) else "-"
            lw = 1.5 if arm in ("gpt", "claude", "gemini") else 1.0
            ax.plot(ks_arm, means, color=color, lw=lw, ls=ls, marker="o",
                    ms=2.6, label=label)
            if arm != "gemini" or min(ns) >= 3:
                ax.fill_between(ks_arm, los, his, color=color, alpha=0.13, lw=0)
            if arm == "gemini":
                note(f"[fig3] gemini n per k: {dict(zip(ks_arm, ns))}")
            if arm == "gpt":
                note(f"[fig3] gpt dose {col}: " + ", ".join(
                    f"k{k}={m:.3f}" for k, m in zip(ks_arm, means)))
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.minorticks_off()
        ax.set_xlabel("Exposure $L$ (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7)

    ax = fig.add_subplot(gs[:, 0])
    ax.axvspan(10, 15, color=C_CLAUDE, alpha=0.10, zorder=0)
    dose(ax, "kappa_pref_top3", r"Final $\kappa$", "Concentration vs. exposure", "a")
    fig.legend(*ax.get_legend_handles_labels(), ncol=5, frameon=False,
               fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               columnspacing=1.4, handlelength=1.6)

    ax = fig.add_subplot(gs[0, 1])
    dose(ax, "shipper_surplus", "Shipper surplus", "Welfare mirrors concentration", "b")

    ax = fig.add_subplot(gs[0, 2])
    dose(ax, "service_rate", "Service rate", "LLMs keep the market serving", "c")

    # (d) seed dots k10 vs k15 (GPT)
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d")
    for i, k in enumerate([10, 15]):
        vs = seed_vals("gpt", k, "kappa_pref_top3")
        ax.scatter([i] * len(vs), vs, color=C_GPT, s=14, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.hlines(np.mean(vs), i - 0.18, i + 0.18, color=C_GPT, lw=1.4)
    for s in SEEDS:
        a, b = tail5(f"gpt-k10-s{s}", "kappa_pref_top3"), tail5(f"gpt-k15-s{s}", "kappa_pref_top3")
        ax.plot([0, 1], [a, b], color=C_GRAY, lw=0.5, alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$L=10$", "$L=15$"])
    ax.set_ylabel(r"Final $\kappa$")
    ax.set_xlim(-0.5, 1.5)
    ax.set_title("The jump, seed by seed (GPT)", fontsize=7)
    v10, v15 = seed_vals("gpt", 10, "kappa_pref_top3"), seed_vals("gpt", 15, "kappa_pref_top3")
    note(f"[fig3d] gpt k10 {[round(v,2) for v in v10]} vs k15 {[round(v,2) for v in v15]}; "
         f"all-seeds-increase={all(b > a for a, b in zip(v10, v15))}")

    # (e) GPT time series by k
    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "e")
    for k in KS:
        ys = np.array([rounds(f"gpt-k{k}-s{s}")["kappa_pref_top3"].values
                       for s in SEEDS])
        ax.plot(range(1, 31), ys.mean(0), color=K_RAMP[k],
                lw=1.5 if k >= 15 else 0.9, label=f"$L={k}$")
    ax.set_xlabel("Day")
    ax.set_ylabel(r"Concentration $\kappa$")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=5.5,
              columnspacing=0.8, handlelength=1.2)
    ax.set_title("Daily paths by exposure (GPT)", fontsize=7)
    save(fig, "fig3")


# ====================================================================
# Fig 4 — information structure and unpredictability
# ====================================================================
def mode_vals(arm):
    """Final kappa per run; all three trust-signal conditions get 10 runs."""
    n = 10 if arm in ("gpt-static", "gpt", "gpt-truth") else 5
    vs = [tail5(f"{arm}-k20-s{s}", "kappa_pref_top3") for s in range(n)]
    return [v for v in vs if v is not None]


def fig4():
    fig = plt.figure(figsize=(W, 48 * MM))
    gs = fig.add_gridspec(1, 4, wspace=0.45, left=0.07,
                          right=0.99, top=0.88, bottom=0.19)
    modes = [("gpt-truth", "Truth", C_GRAY),
             ("gpt-static", "Static", C_CLAUDE),
             ("gpt", "Endogenous", C_GPT)]

    # (a) bars + dots (no title)
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    for i, (arm, label, color) in enumerate(modes):
        vs = mode_vals(arm)
        ax.bar(i, np.mean(vs), width=0.62, color=color, alpha=0.85)
        ax.scatter([i] * len(vs), vs, color="k", s=9, zorder=3,
                   edgecolors="white", linewidths=0.5)
        note(f"[fig4a] {label} n={len(vs)} mean={np.mean(vs):.3f} "
             f"sd={np.std(vs, ddof=1):.3f} "
             f"range=[{min(vs):.3f},{max(vs):.3f}]")
    ax.set_xticks(range(3))
    ax.set_xticklabels([m[1].lower() for m in modes], fontsize=6.5)
    ax.set_ylabel(r"Final $\kappa$ ($L=20$)")
    ax.set_ylim(0.28, 0.72)

    # (b–d) spaghetti per structure
    for idx, (arm, label, color) in enumerate(modes):
        ax = fig.add_subplot(gs[0, idx + 1])
        panel(ax, "bcd"[idx], dx=-0.30)
        n_seeds = 10 if arm in ("gpt-static", "gpt", "gpt-truth") else 5
        for s in range(n_seeds):
            df = rounds(f"{arm}-k20-s{s}")
            if df is None:
                continue
            ax.plot(range(1, 31), df["kappa_pref_top3"], color=color,
                    lw=0.7, alpha=0.75)
        ax.set_ylim(0.15, 1.0)
        ax.set_xlabel("Day")
        if idx == 0:
            ax.set_ylabel(r"Concentration $\kappa$")
        ax.set_title(label, fontsize=7)
    save(fig, "fig4")


# ====================================================================
# Fig 5 — intervention race
# ====================================================================
def fig5():
    from scipy import stats as sps
    fig = plt.figure(figsize=(W, 55 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.07,
                          right=0.99, top=0.86, bottom=0.17)
    arms = [("gpt", "No intervention", C_GPT, "o"),
            ("showcap", "Capacity disclosure", C_GEMINI, "o"),
            ("noshuffle", "Fixed list order", C_PURPLE, "s"),
            ("popularity", "Popularity display", C_GREEDY, "^"),
            ("poly3", "Vendor mix", C_CLAUDE, "D")]

    def arm_vals(arm, col):
        n = 10 if arm == "gpt" else 5
        vs = [tail5(f"{arm}-k20-s{s}", col) for s in range(n)]
        return [v for v in vs if v is not None]

    # (a) kappa bars + Mann-Whitney vs none (biology-style brackets)
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a")
    base = arm_vals("gpt", "kappa_pref_top3")
    tops = {}
    for i, (arm, label, color, m) in enumerate(arms):
        vs = arm_vals(arm, "kappa_pref_top3")
        ax.bar(i, np.mean(vs), width=0.6, color=color, alpha=0.85)
        ax.scatter([i] * len(vs), vs, color="k", s=8, zorder=3,
                   edgecolors="white", linewidths=0.4)
        tops[i] = max(vs)
    # significance brackets from 'none' (x=0) to each comparison arm
    GRAYINK = "#333333"
    y0 = max(tops.values()) + 0.04
    for j, (arm, label, color, m) in enumerate(arms):
        if arm == "gpt":
            continue
        vs = arm_vals(arm, "kappa_pref_top3")
        U, p = sps.mannwhitneyu(vs, base, alternative="two-sided")
        txt = f"$p$={p:.2f}" if p >= 0.001 else "$p$<0.001"
        h = y0 + 0.055 * (j - 1)
        ax.plot([0, 0, j, j], [h - 0.012, h, h, tops[j] + 0.02],
                lw=0.7, color=GRAYINK, clip_on=False)
        ax.text(j / 2, h + 0.010, txt, ha="center", fontsize=5.5,
                color=GRAYINK)
        note(f"[fig5a] {label} vs none: U={U:.0f} p={p:.4f}")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["None", "Capacity", "Order", "Popular.", "Mix"],
                       fontsize=6)
    ax.set_ylabel(r"Final $\kappa$")
    ax.set_ylim(0, 0.88)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_title("Concentration by intervention", fontsize=7)

    # (b) daily request HHI: showcap vs base
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    for arm, color, label in [("gpt", C_GPT, "No intervention"),
                              ("showcap", C_GEMINI, "Capacity disclosure")]:
        allh = []
        for s in SEEDS:
            day_votes = {}
            for r in trace_iter(f"{arm}-k20-s{s}"):
                if r.get("type") == "decision":
                    day_votes.setdefault(r["round"], []).append(r["ranked"][0])
            hh = []
            for t in sorted(day_votes):
                c = Counter(day_votes[t])
                sh = np.array(list(c.values())) / 50
                hh.append(float((sh ** 2).sum()))
            allh.append(hh)
        allh = np.array(allh)
        ax.fill_between(range(1, 31), allh.min(0), allh.max(0), color=color,
                        alpha=0.15, lw=0)
        ax.plot(range(1, 31), allh.mean(0), color=color, lw=1.2, label=label)
        note(f"[fig5b] {label}: day1 HHI={allh[:,0].mean():.2f}, "
             f"final HHI={allh[:,-1].mean():.2f}")
    ax.set_xlabel("Day")
    ax.set_ylabel("HHI of daily requests")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("Disclosure disperses demand upstream", fontsize=7)

    # (c) welfare frontier
    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c")
    for arm, label, color, marker in arms:
        ka = arm_vals(arm, "kappa_pref_top3")
        su = arm_vals(arm, "shipper_surplus")
        ax.scatter(ka, su, color=color, marker=marker, s=20, label=label,
                   edgecolors="white", linewidths=0.5, zorder=3)
        note(f"[fig5c] {label}: kappa mean={np.mean(ka):.2f} "
             f"[{min(ka):.2f},{max(ka):.2f}] surplus mean={np.mean(su):.2f} (n={len(ka)})")
    ax.set_xlabel(r"Final $\kappa$")
    ax.set_ylabel("Shipper surplus")
    ax.legend(frameon=False, fontsize=5.5, loc="upper right",
              handletextpad=0.2)
    ax.set_title("Concentration vs. shipper surplus", fontsize=7)
    save(fig, "fig5")


# ====================================================================
# Appendix figures
# ====================================================================
def figA1():
    """Appendix B: all four reference rules across exposure."""
    fig = plt.figure(figsize=(W, 50 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.35, left=0.08, right=0.98,
                          top=0.86, bottom=0.18)
    rules = [("random", C_GRAY, "Random"),
             ("pa", "#8a6fd1", "Preferential attachment"),
             ("conflicting", "#c05a9e", "Conflicting attachment"),
             ("greedy", C_GREEDY, "Common-score\n(deterministic)")]
    for idx, (col, ylabel, title) in enumerate([
            ("kappa_pref_top3", r"Final $\kappa$", "Concentration"),
            ("service_rate", "Service rate", "Service")]):
        ax = fig.add_subplot(gs[0, idx])
        panel(ax, "ab"[idx], dy=1.14)
        for arm, color, label in rules:
            means, los, his = [], [], []
            for k in KS:
                vs = seed_vals(arm, k, col)
                means.append(np.mean(vs)); los.append(min(vs)); his.append(max(vs))
            ax.plot(KS, means, color=color, lw=1.2, marker="o", ms=2.5,
                    label=label)
            ax.fill_between(KS, los, his, color=color, alpha=0.13, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.minorticks_off()
        ax.set_xlabel("Exposure $L$ (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7)
        if idx == 0:
            ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    save(fig, "figA1")


def figA2():
    """Appendix C: alternative concentration measures (GPT)."""
    fig = plt.figure(figsize=(W, 50 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.07, right=0.99,
                          top=0.86, bottom=0.18)
    metrics = [("kappa_top3", r"Final realized $\kappa$", "Realized allocation"),
               ("revenue_top3_share", "Top-3 revenue share", "Carrier revenue"),
               ("hhi_round", "HHI of served loads", "Served-load HHI")]
    for idx, (col, ylabel, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        panel(ax, "abc"[idx], dy=1.14)
        for arm, color, label in [("gpt", C_GPT, "GPT"),
                                  ("random", C_GRAY, "Random")]:
            means, los, his = [], [], []
            for k in KS:
                vs = seed_vals(arm, k, col)
                means.append(np.mean(vs)); los.append(min(vs)); his.append(max(vs))
            ls = "--" if arm == "random" else "-"
            ax.plot(KS, means, color=color, lw=1.3, ls=ls, marker="o", ms=2.5,
                    label=label)
            ax.fill_between(KS, los, his, color=color, alpha=0.13, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.minorticks_off()
        ax.set_xlabel("Exposure $L$ (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7)
        if idx == 0:
            ax.legend(frameon=False, fontsize=6)
    save(fig, "figA2")


# ====================================================================
# Fig 6 — settled matching network (preference type -> carrier)
# ====================================================================
TYPE_NAMES = ["Cost", "Reliability", "Capacity", "Specialty", "Balanced"]
TYPE_COLORS = [C_ACC, C_GPT, C_GEMINI, C_PURPLE, "#8a8a8a"]


def fig6():
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    cmap = LinearSegmentedColormap.from_list("r", ["#f4e3c1", "#1a7a4a"])

    cases = [("gpt-k20-s0", "Default run"),
             ("gpt-k20-s9", "Locked-in run"),
             ("showcap-k20-s0", "Capacity disclosure")]

    fig = plt.figure(figsize=(W, 64 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.055, right=0.90,
                          top=0.90, bottom=0.10)

    for ci, (tag, title) in enumerate(cases):
        ax = fig.add_subplot(gs[ci])
        panel(ax, "abc"[ci], dx=-0.10)
        # final-week first choices by preference type
        cnt = np.zeros((5, 20))
        final_state = None
        for r in trace_iter(tag):
            if r.get("type") == "decision" and r["round"] >= 26:
                cnt[r["shipper"] % 5, r["ranked"][0]] += 1
            elif r.get("type") == "round_state" and r["round"] == 30:
                final_state = {c["carrier"]: c for c in r["carriers"]}
        kappa = tail5(tag, "kappa_pref_top3")
        # carriers sorted by final unit price (cheap at bottom)
        order = sorted(range(20), key=lambda j: final_state[j]["price"])
        ypos = {j: i / 19 for i, j in enumerate(order)}
        ty = {t: 0.08 + 0.84 * t / 4 for t in range(5)}
        # edges
        for t in range(5):
            for j in range(20):
                if cnt[t, j] > 0:
                    ax.plot([0, 1], [ty[t], ypos[j]], color=TYPE_COLORS[t],
                            lw=0.25 * cnt[t, j], alpha=0.55,
                            solid_capstyle="round", zorder=1)
        # type nodes
        for t in range(5):
            ax.scatter([0], [ty[t]], s=42, marker="s", color=TYPE_COLORS[t],
                       zorder=3)
            if ci == 0:
                ax.text(-0.06, ty[t], TYPE_NAMES[t], ha="right", va="center",
                        fontsize=6, color=TYPE_COLORS[t])
        # carrier nodes
        tot = cnt.sum(0)
        n_zero = int((tot == 0).sum())
        for j in range(20):
            if tot[j] == 0:
                ax.scatter([1], [ypos[j]], s=14, facecolors="white",
                           edgecolors="#9a9a9a", linewidths=0.7,
                           linestyle="--", zorder=3)
            else:
                sc = ax.scatter([1], [ypos[j]], s=16 + tot[j] * 3.2,
                                c=[final_state[j]["rating"]], cmap=cmap,
                                vmin=0, vmax=1, edgecolors="white",
                                linewidths=0.5, zorder=3)
        ax.set_title(f"{title} ($\\kappa={kappa:.2f}$)", fontsize=7)
        ax.set_xlim(-0.08, 1.10)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")
        ax.text(1.075, 0.5,
                "Carriers by final unit price (cheap $\\rightarrow$ expensive)",
                rotation=90, va="center", fontsize=5.5, color="#555555")
        # digest stats
        w = cnt / cnt.sum(1, keepdims=True)
        price = np.array([final_state[j]["price"] for j in range(20)])
        rel = {c["carrier"]: c for c in
               next(trace_iter(tag))["carriers"]}
        true_rel = np.array([rel[j]["reliability"] for j in range(20)])
        note(f"[fig6:{tag}] kappa={kappa:.2f} zero-demand carriers={n_zero} "
             f"| mean price chosen: cost={w[0] @ price:.2f} "
             f"reliab={w[1] @ price:.2f} "
             f"| mean true-rel chosen: cost={w[0] @ true_rel:.2f} "
             f"reliab={w[1] @ true_rel:.2f}")

    cb = fig.colorbar(sc, ax=fig.axes, fraction=0.022, pad=0.015)
    cb.set_label("Final displayed rating", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([], [], color="#9a9a9a", lw=0, marker="o",
                               markerfacecolor="white", markeredgecolor="#9a9a9a",
                               markersize=4.5, label="No demand in final week")],
               frameon=False, fontsize=6, loc="lower right",
               bbox_to_anchor=(0.89, 0.005))
    save(fig, "fig6")


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); figA1(); figA2()
    (OUT / "stats_digest.txt").write_text("\n".join(DIGEST), encoding="utf-8")
    print("digest written")


if __name__ == "__main__":
    main()
