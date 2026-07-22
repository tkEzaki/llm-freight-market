# When Shippers Become Algorithms

Simulation platform and experiment code for the paper:

> *When Shippers Become Algorithms: Candidate Exposure, Information Design,
> and the Concentration of LLM-Mediated Freight Markets.*

Fifty LLM shipper agents procure truckload service from twenty carriers over
thirty days in a market with waterfall tendering, binding daily capacity,
congestion pricing, and an endogenous customer-rating system. The platform
supports OpenAI, Anthropic, and Google models (plus several other backends)
and four non-LLM reference decision rules.

## Repository layout

```
src/
├── runner.py            # one experimental cell: CellConfig -> rounds CSV + trace JSONL
├── economy.py           # capacity, congestion pricing, ratings, welfare accounting
├── agents/
│   ├── shipper_llm.py   # LLM shipper agents (prompt construction, priorities)
│   ├── carriers.py      # carrier attributes
│   └── baselines.py     # random / preferential / conflicting attachment / common-score
├── llm/                 # backend abstraction (OpenAI / Anthropic / Google / ...),
│                        #   JSON response parser, disk cache, fail-fast quota handling
└── network/             # bipartite bookkeeping and concentration metrics
experiments/
├── run_k_sweep2.py      # all experimental phases reported in the paper (resumable)
└── make_paper_figs.py   # reproduces every figure and the numeric digest
data/k_sweep2/           # complete results of all reported runs:
                         #   rounds_<cell>.csv      per-day market aggregates
                         #   trace_<cell>.jsonl.gz  decision-level audit logs
                         #   (every prompt, raw model response, parsed
                         #    ranking, and daily market state)
```

### Naming of cells

Filenames encode a cell as `<condition>-k<L>-s<run>`, e.g. `gpt-k20-s0` is
GPT with twenty displayed candidates, run 0. The `k` in these tags is the
exposure lever the paper calls `L` (the code predates the rename); the
number after it is the number of candidates shown per load. Conditions are
`gpt` / `claude` / `gemini` (LLM vendors), `gpt-truth` / `gpt-static`
(trust-signal ablation; plain `gpt` is the endogenous rating), `showcap`
(capacity disclosure), `noshuffle` (fixed list order), `popularity`
(popularity display), `poly3` (vendor mix), `gpt-pr05` / `gpt-pr02`
(slower price adjustment), and `random` / `pa` / `conflicting` / `greedy`
(the four reference decision rules).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # then fill in your API keys
```

Experiments call commercial LLM APIs. The full study makes ~190k calls
(≈ $200 at 2026 prices); a single cell (50 shippers × 30 days) makes 1,500
calls and takes about five minutes at 16-way concurrency.

## Running

```bash
# one phase (e.g. the GPT exposure sweep); resumable, completed cells are skipped
python -X utf8 experiments/run_k_sweep2.py v2

# everything reported in the paper
python -X utf8 experiments/run_k_sweep2.py

# figures + stats digest (results/ if present, else the shipped data/)
python -X utf8 experiments/make_paper_figs.py
```

Outputs go to `results/k_sweep2/`: per-day market aggregates
(`rounds_<cell>.csv`) and a decision-level audit trail
(`trace_<cell>.jsonl`) containing every prompt, raw model response, parsed
ranking, and end-of-day market state. Prompt-level response caching makes
every cell exactly reproducible. Quota and billing failures abort the run
immediately so that random fallbacks never contaminate the data; a
transient API error is retried, and a run also aborts if more than 5% of
its decisions ever fall back to a random choice (none did in the
reported runs).

## Provenance of the reported runs

The reported experiments called the API model identifiers
`gpt-5.4-mini` (OpenAI), `claude-haiku-4-5` (Anthropic), and
`gemini-3.5-flash` (Google) between 15 and 18 July 2026, at temperature
0.7 with a 256-token output limit, reasoning modes disabled, and three
retries with exponential backoff. One cell (50 shippers × 30 days) makes
1,500 calls and completes in about five minutes at 16-way concurrency;
the capacity-disclosure condition runs sequentially within each day,
because the disclosed remaining capacity depends on decisions taken
earlier that day, and takes about 45 minutes per cell. Total API cost for
all reported experiments was on the order of $200.

A response counts as valid if it yields three distinct carrier
identifiers drawn from the displayed pool (duplicates and non-candidates
are dropped during parsing). Across the ~190,000 decisions reported in
the paper, every response met this check, none returned fewer than three
usable choices, and no decision fell back to a non-LLM rule.

Because providers update hosted models, the cached prompt–response pairs
under `data/` replay the recorded behavior exactly, while fresh API calls
to the same identifiers may differ.

## Reproducing the paper's figures without API calls

The repository ships the complete data of all reported cells
(~190k LLM decisions) under `data/k_sweep2/`. The figure script reads
fresh runs from `results/` when present and falls back to the shipped
`data/` (gzipped traces are read transparently), so

```bash
python -X utf8 experiments/make_paper_figs.py
```

reproduces every figure and the numeric digest from the shipped data
alone, with no API access.

## License

MIT — see `LICENSE`.
