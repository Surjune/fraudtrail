"""Collect the evidence one investigation needs, recording every call made.

The call log becomes `tool_calls` in the answer file and the CaseEvent trail in the graph,
so an analyst can see exactly which queries the conclusion rests on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from fraudtrail.casepack import ExamCase
from fraudtrail.evidence.models import CaseEvidence
from fraudtrail.evidence.provider import EvidenceProvider
from fraudtrail.investigate import constants as c


@dataclass(frozen=True)
class ToolCall:
    name: str
    params: dict[str, Any]
    result_size: int


@dataclass(frozen=True)
class Gathered:
    evidence: CaseEvidence
    calls: tuple[ToolCall, ...]

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _query_text(case: ExamCase) -> str:
    """What this case is about, in the analysts' own vocabulary, for memory retrieval."""
    return f"{case.trigger_text} card {case.card_id} customer {case.customer_id}"


def gather(provider: EvidenceProvider, case: ExamCase) -> Gathered:
    calls: list[ToolCall] = []

    def record(name: str, params: dict[str, Any], size: int) -> None:
        calls.append(ToolCall(name=name, params=params, result_size=size))

    flagged = provider.flagged_txn(case.flagged_txn_id)
    record("flagged_txn", {"txn_id": case.flagged_txn_id}, 1)

    # Episodes reach back days, and card testing reaches back weeks.
    lookback = timedelta(days=max(c.EPISODE_LOOKBACK_DAYS, c.CARD_TESTING_LOOKBACK_DAYS))
    window_start = case.opened_at - lookback
    window = provider.card_window(case.card_id, window_start, case.opened_at)
    record(
        "card_window",
        {
            "card_id": case.card_id,
            "from": window_start.isoformat(),
            "to": case.opened_at.isoformat(),
        },
        len(window),
    )

    baseline = provider.card_baseline(case.card_id, flagged.ts)
    record(
        "card_baseline",
        {"card_id": case.card_id, "before": flagged.ts.isoformat()},
        baseline.n_txns,
    )

    regions = provider.region_timeline(case.card_id, case.opened_at)
    record("region_timeline", {"card_id": case.card_id}, len(regions))

    prior = provider.prior_cases(case.card_id, case.opened_at)
    record("prior_cases", {"card_id": case.card_id}, len(prior))

    ring_start = case.opened_at - timedelta(days=c.RING_WINDOW_DAYS)
    reach = None
    if flagged.profile_id:
        reach = provider.device_reach(flagged.profile_id, ring_start, case.opened_at)
        record("device_reach", {"profile_id": flagged.profile_id}, len(reach.cards))

    shared = provider.shared_origins(
        case.card_id, ring_start, case.opened_at, c.SHARED_ORIGIN_MIN_CARDS
    )
    record(
        "shared_origins",
        {"card_id": case.card_id, "min_cards": c.SHARED_ORIGIN_MIN_CARDS},
        len(shared),
    )

    recurring = provider.recurring_matches(
        case.card_id, flagged.amount, flagged.product_cd, c.RECURRING_TOLERANCE_USD, case.opened_at
    )
    record(
        "recurring_charges",
        {"card_id": case.card_id, "amount": flagged.amount, "tolerance": c.RECURRING_TOLERANCE_USD},
        len(recurring),
    )

    similar = provider.similar_cases(_query_text(case), case.opened_at)
    record("similar_closed_cases", {"query": _query_text(case)}, len(similar))

    cards = provider.customer_cards(case.card_id)
    record("customer_cards", {"card_id": case.card_id}, len(cards))

    evidence = CaseEvidence(
        flagged=flagged,
        card_id=case.card_id,
        customer_id=case.customer_id,
        opened_at=case.opened_at,
        window=window,
        baseline=baseline,
        device_reach=reach,
        regions=regions,
        prior_cases=prior,
        shared_origins=shared,
        customer_cards=cards,
        recurring_candidates=recurring,
        similar_cases=similar,
    )
    return Gathered(evidence=evidence, calls=tuple(calls))
