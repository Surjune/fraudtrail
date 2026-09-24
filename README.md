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
| Verdicts | 11 fraud, 9 legitimate |
| Patterns found | 7 card-not-present from a new device, 1 card-not-present, 1 account takeover, **2 undocumented** |
| Reports filed | 3 |
| Cases that asked for more evidence | 10, and the recommendation changed in all 10 |
| Approval routes recommended | 36 auto, 11 L1, 3 L2 |
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

**As case memory.** Each finished investigation is written back as an `InvestigationCase`
connected to its card, its transactions, the devices and the prior cases it drew on, with every
step — each evidence query, the assessment, the decision before evidence was requested, the
response, the decision after it — stored in order as `CaseEvent` vertices. A case can be
replayed, not just inspected.

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
| `src/fraudtrail/graph/` | TigerGraph client and case write-back |
| `src/fraudtrail/graphrag/` | The document corpus and local embeddings |
| `src/fraudtrail/answer/` | Answer schema, validation, and the narrators |
| `src/fraudtrail/llm/` | Three providers over plain HTTP, no SDKs |
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

`--only HHG-014` runs one case, `--out DIR` writes elsewhere, and `--offline` takes the
evidence from the local DuckDB warehouse instead of the graph, which needs no workspace at all.

Checks:

```bash
uv run ruff check . && uv run mypy --strict src scripts && uv run pytest
```

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
- **No accuracy measurement against an answer key**, because the key is not public. The checks
  above are for consistency and policy compliance, not for correctness.
- **The regulatory references are cited but not ingested.** The corpus holds the policy, the
  patterns and the reference list from the dataset guide; the linked FinCEN and FATF PDFs are
  not downloaded.
- **Card identifiers are derived**, not given. The dataset has no card column, so cards are
  reconstructed per customer from the issuer fields; the rule and how it was verified are in
  `docs/data_findings.md`.
- **No monitoring mode.** The agent investigates the twenty cases it is given; it does not sweep
  the exam window for alerts of its own.
