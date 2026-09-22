# FraudTrail

An agent that investigates card fraud on TigerGraph, progresses a case as evidence arrives, and
recommends the next best action under the bank's fraud policy. Built for the TigerGraph ×
Hacker House Goa challenge on the HHGOA_IEEE dataset.

## Design in one paragraph

Graph analysis (GSQL queries and TigerGraph graph algorithms) finds the evidence. Deterministic
code turns that evidence into a decision: the policy engine applies rules R1–R10, the approval
routes, the case-versus-report rule and the stopping rule exactly as written. The LLM selects
tools, synthesizes evidence retrieved through GraphRAG, and writes the explanations and the
suspicious activity report. Every answer file passes a validator before it is written.

## Layout

| Path | What it holds |
| --- | --- |
| `src/fraudtrail/policy/` | Fraud Policy v1.0 as code: actions, routes, thresholds, rules |
| `src/fraudtrail/answer/` | Answer-file schema and validation |
| `scripts/profile_data.py` | Dataset profiling that the graph design depends on |
| `cases/` | One answer file per exam case |
| `tests/` | Unit tests |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

Put the five dataset files (`README.md`, `transactions.csv`, `identity.csv`,
`closed_cases_history.csv`, `case_pack.csv`) in `data/raw/`. The folder is git-ignored.

```bash
uv run python scripts/profile_data.py
```

Copy `.env.example` to `.env` and fill in the TigerGraph Savanna and LLM credentials.

## Known limitations

Tracked here as the build progresses.
