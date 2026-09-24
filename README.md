# FraudTrail

An agent that investigates card fraud on TigerGraph: it takes an alert, gathers evidence from
the graph, works out what kind of fraud it is and how far it reaches, asks for more evidence
when what it has does not settle the question, and recommends what the bank should do and who
must approve it. Every finished case is written back into the graph as memory for the next one.

Built for the TigerGraph × Hacker House Goa challenge on the HHGOA_IEEE dataset.

## The idea in one paragraph

The graph finds the evidence and code makes the decisions. Seventeen installed GSQL queries do
the graph's half of the work: ten answer the questions an investigator asks — what else
happened on this card, which other cards share this device, what did the bank decide last time
— and the rest search the vector store and write the finished case back. A policy engine turns
what comes back into a verdict, an action and an approval route by applying the Fraud Policy's
rules exactly as written. A language model writes the prose and nothing else: every identifier, amount and date
it returns is checked against the evidence, and anything invented is thrown away. Nothing in an
answer file rests on a model's recollection.

## What it produces

Twenty answer files in [`cases/`](cases), one per exam case, each validated before it is
written. From the current run:

| | |
| --- | --- |
| Verdicts | 6 fraud, 14 legitimate |
| Patterns found | 3 card-not-present from a new device, 1 account takeover, **2 undocumented** |
| Reports filed | 2 |
| Cases that asked for more evidence | 10, and the recommendation changed in all 10 |
| Approval routes recommended | 46 auto, 6 L1, 2 L2 |
| Evidence queries per case | 10–11 |
| Cases written to the graph | 20 of 20, with 323 case events between them |

The two undocumented findings are ones the bank's own closed cases describe but never named.
**HHG-014** is a shared-device ring: one fully specified device profile used on 20 cards across
20 customers in the 30 days before the alert, new to every account it touched. **HHG-006** is a
burst of four online purchases just under a $500 limit, $1,906.07 in about forty minutes, each
small enough that the bank's model stayed quiet.

## How TigerGraph is used

**As the evidence store.** 590,742 transactions, 14,318 cards, 13,553 customers, 9,705 device
profiles, 5,565 closed cases and 2.5M edges. The interesting joins are edges: a card to
its transactions, a transaction to the device profile behind it, a closed case to the
transactions it covered and the cards it named as connected.

**As the query engine.** Every question is an installed GSQL query, so the traversal runs where
the data is. `device_reach` is the one that matters most: given a device profile and a window
it returns the cards, the customers, the transactions and any confirmed-fraud cases behind it,
plus the share of those transactions the bank marked new for their own account. That share is
what separates a ring from a browser fingerprint hundreds of unrelated customers happen to
share. It answers in about a tenth of a second.

**As the vector store.** Every closed-case note and every section of the policy, the documented
patterns and the regulatory references carry a 384-dimension embedding. Retrieval is a
`vectorSearch` followed by a graph hop onto the card each hit was opened on, so a retrieved
case arrives connected to what it touched rather than as a loose paragraph.

**Through MCP.** `run_agent.py --mcp` reaches the graph through the TigerGraph MCP
server instead of REST: the same installed queries, called as MCP tools over stdio, with
the same answers out the other end. That is the path an external agent framework would
take, and `src/fraudtrail/graph/mcp_client.py` is the client for it.

**As case memory.** Each finished investigation is written back as an `InvestigationCase`
connected to its card, its transactions, the devices and the prior cases it drew on, with every
step — each evidence query, the assessment, the decision before evidence was requested, the
response, the decision after it — stored in order as `CaseEvent` vertices. A case can be
replayed, not just inspected.

## The interface

```bash
uv run streamlit run app/dashboard.py
```

One case at a time, read back out of the graph rather than from the answer files, in five
tabs: how the case progressed step by step, the evidence it rests on, how uncertain the
agent was and what it asked for, what it recommends before and after that evidence with
the approval each action needs, and the report when policy calls for one.

Two other views: an **approval queue** of everything across the twenty cases that a human
must sign off (8 actions: 6 at L1, 2 at L2), and a **ring finder** that runs label
propagation in the graph over cards sharing a device profile, so the communities come out
of the algorithm rather than out of a list someone wrote.

## Agentic behaviour

- **It decides what to look at.** What it gathers depends on what the first pass found: a
  device profile is only expanded when the flagged transaction has one, and case memory is
  searched a second time using what the detectors found rather than the alert's wording.
- **It knows when it cannot decide.** Policy section 5 and rules R1 and R4 govern when to ask;
  ten cases asked for customer validation or step-up authentication rather than acting on a
  single signal.
- **It changes its mind.** The answer records the recommendation before evidence was requested
  and after the response arrived. In all ten cases the recommendation changed.
- **It stops.** Section 6's stopping rule ends the investigation when a response settles the
  question, or when the probability is decisive on two independent pieces of evidence, and the
  answer says which rule stopped it.
- **It knows what it may not do.** Every action carries its approval route: `auto`, `L1` or
  `L2`. Blocking a card above $2,500 and filing a report are always L2. The agent recommends;
  it never claims to have executed anything a human must approve.
- **It remembers.** Cases run oldest first, so each one can retrieve both the bank's closed
  cases and the investigations this agent finished earlier.

## Why the decisions are code, not a prompt

The Fraud Policy is a specification: fourteen actions, three approval routes, ten rules,
explicit thresholds. Rules R1 to R10, the case-versus-report rule, the exposure definition and
the stopping rule live in `src/fraudtrail/policy/`, where every threshold is a named constant
citing the rule it comes from. A model that mostly follows a policy is a model that
occasionally does not, and "occasionally" is a blocked card that should not have been blocked.

The model is an upgrade, not a dependency. With `FRAUDTRAIL_LLM_PROVIDER=none` the agent still
produces twenty complete, valid answer files; the prose comes from templates instead.

## Layout

| Path | What it holds |
| --- | --- |
| `src/fraudtrail/policy/` | The Fraud Policy as code: actions, routes, thresholds, rules R1–R10 |
| `src/fraudtrail/investigate/` | Detectors, scoring, evidence gathering, the investigation loop |
| `src/fraudtrail/evidence/` | One interface, two sources: TigerGraph and the local warehouse |
| `src/fraudtrail/graph/` | TigerGraph client, the MCP client, and case write-back |
| `src/fraudtrail/graphrag/` | The document corpus and local embeddings |
| `src/fraudtrail/answer/` | Answer schema, validation, and the narrators |
| `src/fraudtrail/llm/` | Three providers over plain HTTP, no SDKs |
| `src/fraudtrail/evaluate/` | Replaying closed cases and scoring the agent against them |
| `app/dashboard.py` | The analyst interface |
| `graph/` | Schema, vector attributes, loading job, and the installed queries |
| `scripts/` | Profile, export, load, install, embed, run |
| `cases/` | One answer file per exam case |
| `docs/data_findings.md` | What profiling established about the dataset |

## Running it

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Put the five dataset files
(`README.md`, `transactions.csv`, `identity.csv`, `closed_cases_history.csv`, `case_pack.csv`)
in `data/raw/`, which is git-ignored.

```bash
uv sync
cp .env.example .env     # then fill in TG_HOST and TG_SECRET
```

Build the local warehouse and the files the graph is loaded from:

```bash
uv run python scripts/profile_data.py
uv run python scripts/export_graph.py
```

Create the graph, load it, install the queries, and embed the corpus:

```bash
uv run python scripts/load_graph.py
uv run python scripts/install_queries.py
uv run python scripts/build_embeddings.py
```

Run the agent over all twenty cases:

```bash
uv run python scripts/run_agent.py
```

`--only HHG-014` runs one case, `--out DIR` writes elsewhere, `--mcp` reaches the graph
through the TigerGraph MCP server, and `--offline` takes the evidence from the local DuckDB
warehouse instead of the graph, which needs no workspace at all.

Measure it against the bank's closed cases, and open the interface:

```bash
uv run python scripts/evaluate.py --n 120
uv run streamlit run app/dashboard.py
```

Checks:

```bash
uv run ruff check . && uv run mypy --strict src scripts && uv run pytest
```

## Measured accuracy

The exam's answer key is not public, so accuracy is measured where truth is written down:
the bank's 5,565 closed cases. `scripts/evaluate.py` replays them as the alerts they
started as, with the case under test hidden from every query so the agent cannot retrieve
its own answer, and compares what it concluded with what the analysts concluded. Sampling
is stratified and seeded, so a change in the score is a change in the agent.

Over 120 replayed cases:

| Measure | Result |
| --- | --- |
| Verdict accuracy | **85.0%** |
| Confirmed fraud called fraud | 80.0% |
| Cleared alerts cleared | 90.0% |
| Pattern named correctly | 56.2% |
| Suspicious transactions found | 47.4% |
| Exposure matched exactly | 38.3% |
| Action agreement (F1) | 58.5% |
| Report decision agreement | 87.5% |

The full report, including the pattern confusion matrix, is in
[`docs/evaluation.md`](docs/evaluation.md).

The harness earned its keep immediately. Verdict accuracy started at 56.7%, and the reason
was not the investigation but the simulated customer reply: alerts the analysts had cleared
were being denied by the assumed cardholder and so came out as fraud. The bank's history
says what that reply should be. Where this agent judges the evidence too weak to decide and
asks, the analysts had cleared 76% of those alerts; of the ones it decides without asking,
86% were confirmed fraud. So a denial is now assumed only where the graph establishes
something a cardholder cannot explain away — a device shared across unrelated cards, a
card-testing sequence, or a device already confirmed in another fraud. That single change
took verdict accuracy from 56.7% to 85.0% and cost nothing in fraud recall.

## How it is checked

- **The two evidence sources agree.** The same investigation runs against TigerGraph or against
  a local DuckDB warehouse over the same data. Run both ways, the twenty answers come out
  identical on every scored field, so a future difference points at the evidence rather than at
  the logic.
- **Every answer is validated before it is written.** Wrong approval routes, a report flag that
  disagrees with the policy, a narrative outside six to twelve sentences, a legitimate verdict
  carrying affected transactions, exposure that does not match the transactions it names, or an
  identifier that is not in the dataset — each is an error, and the run reports it rather than
  hiding it. The current run has none.
- **The model cannot introduce a fact.** Its prose is rejected if it contains an identifier,
  amount or date that is not in the evidence it was given, or if it drops one.
- **57 unit tests**, `ruff` and `mypy --strict` clean.

## Known limitations

- **Customer and analyst replies are simulated**, as the round requires. The assumption follows
  the strongest evidence that does not come from the customer, so it cannot manufacture the
  answer the agent wants: where the graph shows the activity is the cardholder's own, the
  simulated cardholder confirms it, even when that closes a case the model leaned towards
  calling fraud. Every assumption is recorded in `evidence_requests`.
- **Episode scoping is the weakest part.** The agent agrees with the analysts on the verdict
  85% of the time but finds only 47% of the transactions they held responsible, and matches
  their exposure figure on 38% of cases. It groups the wrong transactions into an episode more
  often than it reaches the wrong conclusion, which is where the next work belongs.
- **Accuracy is measured against the bank's closed cases, not the exam's answer key**, which
  is not public. A replay resembles the exam but is not it.
- **The regulatory references are cited but not ingested.** The corpus holds the policy, the
  patterns and the reference list from the dataset guide; the linked FinCEN and FATF PDFs are
  not downloaded.
- **Card identifiers are derived**, not given. The dataset has no card column, so cards are
  reconstructed per customer from the issuer fields; the rule and how it was verified are in
  `docs/data_findings.md`.
- **No monitoring mode.** The agent investigates the twenty cases it is given; it does not sweep
  the exam window for alerts of its own.
