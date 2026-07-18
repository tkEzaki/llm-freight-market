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

# figures + stats digest (reads results/k_sweep2/)
python -X utf8 experiments/make_paper_figs.py
```

Outputs go to `results/k_sweep2/`: per-day market aggregates
(`rounds_<cell>.csv`) and a decision-level audit trail
(`trace_<cell>.jsonl`) containing every prompt, raw model response, parsed
ranking, and end-of-day market state. Prompt-level response caching makes
every cell exactly reproducible; runs abort rather than substitute random
choices when an API fails.

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
