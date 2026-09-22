# Data findings

What profiling the HHGOA_IEEE dataset established before any graph or agent work.
Reproduce with `uv run python scripts/profile_data.py`, which writes the full tables to
`reports/data_profile.md`.

## Shape

590,742 transactions (July 2 to December 31, 2016) from 13,553 customers; 144,432
identity records for online transactions; 5,565 closed cases (July 2 to November 2);
20 exam cases (November 11 to December 29). In-person transactions are all product code
`W`; online ones are `C`, `R`, `H` and `S`.

## Derived identifiers

**Card IDs.** `transactions.csv` has `customer_id` but no card ID. A customer's cards are
their distinct (`card4`, `card6`) pairs (network and card type), numbered `K1`, `K2`, …
in ascending order with nulls first. This reproduces the card ID of all 14,975 labelled
transactions in the closed cases and the case pack. 12,793 customers have one card, 755
have two and 5 have three. The rule lives in `src/fraudtrail/data/sql.py`.

**Transaction IDs** are integers written as strings (`"3514030"`), not the `T`-prefixed
form in the README's example.

**Device profiles** are `DeviceInfo | OS | browser | screen` (`id_30`, `id_31`, `id_33`),
with empty positions kept, giving 9,706 distinct profiles. 18% of identity records have
no `DeviceInfo`.

**Billing regions** (`addr1`) are stored as floats (`444.0`); the graph keys them as
integer strings (`"444"`).

## Timing

- An exam case opens exactly six hours after its flagged transaction.
- Every transaction in a closed case's episode happened before the case opened, and
  `first_fraud_txn_id` is always the earliest of them. Investigations look backwards from
  the alert, never forwards.
- Amount offsets added to disguise the data stay within about $0.10: a $100 charge
  appears as $99.94 to $100.09. Recurring-charge matching uses a $0.15 tolerance.

## What the closed cases teach

| Outcome | Pattern | Cases | Actions taken |
| --- | --- | --- | --- |
| cleared | none | 900 | `VERIFY_WITH_CUSTOMER`, `CLOSE_NO_FRAUD` |
| confirmed | card_not_present_fraud | 1,404 | `CREATE_CASE`, `BLOCK_CARD` (+ `FILE_REPORT`) |
| confirmed | account_takeover | 1,205 | same |
| confirmed | card_not_present_new_device | 1,076 | same |
| confirmed | out_of_region_use | 955 | same |
| confirmed | card_testing | 16 | same |
| confirmed | undocumented | 9 | `CREATE_CASE`, `BLOCK_CARD`, `FILE_REPORT` |

**Reports follow the policy exactly.** A report was filed on a confirmed case if and only
if exposure exceeded $1,000 or the pattern was undocumented.

**Cleared alerts have three explanations:** the cardholder confirmed travel to the region
(716), confirmed a purchase from a new phone (158), or confirmed an unusual amount (26).
These are the legitimate stories the agent must be able to recognise.

**The risk score is not the answer.** Cleared cases all had a maximum risk score of 0.8
or above; confirmed fraud appears in every score band, including below 0.1.

**Pattern signatures.** Out-of-region fraud is entirely in-person. Account takeover mixes
in-person and online activity. Card testing episodes are long (median 17 transactions
over about a week). Most other episodes are one or two transactions.

## Undocumented patterns found in analyst notes

1. **Shared-device ring.** One device profile (`SM-G935F Build/NRD90M | Android 7.0 |
   chrome 62.0 for android | 1920x1080`), always behind an anonymous proxy and always
   marked `New`, made two or three online purchases on each of 24 cards from August 15 to
   September 4. The same profile reappears on about 28 further cards from November 14 to
   December 4. Its transactions carry very low risk scores.
2. **Bursts under a $500 limit.** Four online purchases within about forty minutes on one
   card, each just under $500, totalling about $1,900. The notes describe amounts chosen
   to stay under an authorization threshold.

## Exam cases: first observations

- Most exam cards have closed-case history on the same card, and some are repeat
  victims (C09933-K2 has 20 prior confirmed cases). Prior cleared cases matter too: for
  example, C05876-K2 (HHG-012) was previously cleared after the cardholder confirmed
  travel.
- **HHG-014** (analyst request) is the shared-device ring's November wave.
- **HHG-006** (customer report, $482.12) is the flagged end of a burst of four online
  purchases in thirty minutes ($478.95, $456.96, $488.04, $482.12) from two new device
  profiles.
- No exam case is preceded by three or more sub-$5 online authorizations in the hour
  before the flagged transaction.
