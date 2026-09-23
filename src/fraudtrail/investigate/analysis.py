"""Turn evidence into a pattern, an episode and a list of signals.

Candidate patterns each collect support from the detectors and from case memory; the
best-supported one wins. Counter-signals (a trip, a recurring charge, a device the
cardholder confirmed before) are recorded the same way, because half the exam cases are
legitimate and the case file has to say why.
"""

from __future__ import annotations

from dataclasses import replace

from fraudtrail.domain import EvidenceSource, Pattern, Trigger
from fraudtrail.evidence.models import CaseEvidence, PriorCase, SharedOrigin, Txn
from fraudtrail.investigate import constants as c
from fraudtrail.investigate.detectors import (
    Detection,
    Signal,
    card_testing_sequence,
    cnp_burst,
    mixed_channel_anomaly,
    out_of_region_use,
    recurring_charges,
    sublimit_burst,
)

# Phrases the bank's analysts used when they cleared an alert (docs/data_findings.md).
CLEARED_TRAVEL = "confirmed travel"
CLEARED_NEW_PHONE = "new phone"
CLEARED_AMOUNT = "amount unusual"

_RING_DESCRIPTION = (
    "One device profile is being used across many unrelated cards in the same weeks, "
    "always marked new for each account and usually behind an anonymous proxy. The "
    "purchases are ordinary in size, which keeps the bank's model quiet, and the link "
    "between the cardholders is the device rather than anything about the cards."
)
_SUBLIMIT_DESCRIPTION = (
    "Several online purchases on one card inside an hour, each sitting just below a "
    "$500 authorization limit and together far above it. The amounts look chosen to "
    "clear individually rather than to match anything the cardholder usually buys."
)


def _ids(txns: tuple[Txn, ...]) -> tuple[str, ...]:
    return tuple(t.txn_id for t in txns)


def _ring_candidates(evidence: CaseEvidence) -> tuple[SharedOrigin, ...]:
    """Shared device profiles narrow enough to be a ring rather than common software.

    Three tests, all needed: the profile names a real device rather than a partial browser
    fingerprint, it reaches several cards but not hundreds, and it is marked New on nearly
    every account it touches.
    """
    return tuple(
        origin
        for origin in evidence.shared_origins
        if origin.fully_specified
        and c.SHARED_ORIGIN_MIN_CARDS <= origin.card_count <= c.SHARED_ORIGIN_MAX_CARDS
        and origin.new_device_share >= c.SHARED_ORIGIN_MIN_NEW_DEVICE_SHARE
    )


def _memory_signals(evidence: CaseEvidence) -> tuple[list[Signal], dict[str, int], list[str]]:
    signals: list[Signal] = []
    pattern_counts: dict[str, int] = {}
    confirmed: list[str] = []
    cleared_travel: list[str] = []
    cleared_new_phone: list[str] = []

    for case in evidence.prior_cases:
        if case.confirmed_fraud:
            confirmed.append(case.case_id)
            pattern_counts[case.pattern] = pattern_counts.get(case.pattern, 0) + 1
        elif CLEARED_TRAVEL in case.notes:
            cleared_travel.append(case.case_id)
        elif CLEARED_NEW_PHONE in case.notes:
            cleared_new_phone.append(case.case_id)

    if confirmed:
        dominant = max(pattern_counts.items(), key=lambda item: item[1])[0]
        signals.append(
            Signal(
                name="prior_confirmed_fraud",
                family="memory",
                claim=(
                    f"{len(confirmed)} closed case(s) on this card or its owner's cards were "
                    f"confirmed fraud, most often {dominant.replace('_', ' ')}"
                ),
                ref="query:prior_cases(card_id)",
                entity_ids=tuple(confirmed[:5]),
            )
        )
    if cleared_travel:
        signals.append(
            Signal(
                name="prior_cleared_travel",
                family="memory",
                claim=(
                    f"{len(cleared_travel)} earlier alert(s) on this card were cleared when the "
                    f"cardholder confirmed travel to the flagged region"
                ),
                ref="query:prior_cases(card_id)",
                entity_ids=tuple(cleared_travel[:5]),
            )
        )
    if cleared_new_phone:
        signals.append(
            Signal(
                name="prior_cleared_new_phone",
                family="memory",
                claim=(
                    f"{len(cleared_new_phone)} earlier alert(s) on this card were cleared when the "
                    f"cardholder confirmed a purchase from a new phone"
                ),
                ref="query:prior_cases(card_id)",
                entity_ids=tuple(cleared_new_phone[:5]),
            )
        )
    return signals, pattern_counts, confirmed


def _flagged_signals(evidence: CaseEvidence) -> list[Signal]:
    flagged = evidence.flagged
    baseline = evidence.baseline
    signals: list[Signal] = []

    if flagged.risk_score >= 0.7:
        signals.append(
            Signal(
                name="model_score_high",
                family="model",
                claim=(
                    f"The bank's model scored the flagged transaction at {flagged.risk_score:.2f}; "
                    f"above 0.70 most flagged transactions in this book turn out legitimate"
                ),
                ref="attribute:Transaction.risk_score",
                entity_ids=(flagged.txn_id,),
            )
        )
    if flagged.from_new_device:
        signals.append(
            Signal(
                name="new_device",
                family="device",
                claim=(
                    "The identity record marks the device as New for this account "
                    f"({flagged.profile_id or 'no device details'})"
                ),
                ref="query:case_context(flagged_txn)",
                entity_ids=(flagged.txn_id,),
            )
        )
    if flagged.behind_proxy:
        signals.append(
            Signal(
                name="proxy",
                family="device",
                claim=f"The connection used a proxy ({flagged.proxy})",
                ref="query:case_context(flagged_txn)",
                entity_ids=(flagged.txn_id,),
            )
        )
    if baseline is not None and baseline.n_txns >= c.BASELINE_MIN_TXNS:
        z = baseline.amount_z(flagged.amount)
        if z >= c.AMOUNT_UNUSUAL_Z:
            signals.append(
                Signal(
                    name="amount_unusual",
                    family="behaviour",
                    claim=(
                        f"${flagged.amount:,.2f} is {z:.1f} standard deviations above this card's "
                        f"usual ${baseline.mean_amount:,.2f}"
                    ),
                    ref="query:card_baseline(card_id)",
                    entity_ids=(flagged.txn_id,),
                )
            )
        if not baseline.product_seen(flagged.product_cd):
            signals.append(
                Signal(
                    name="product_new",
                    family="behaviour",
                    claim=(
                        f"Product code {flagged.product_cd} has never been used on this card "
                        f"in {baseline.n_txns} prior transactions"
                    ),
                    ref="query:card_baseline(card_id)",
                    entity_ids=(flagged.txn_id,),
                )
            )
        if baseline.profile_seen(flagged.profile_id):
            signals.append(
                Signal(
                    name="device_seen_before",
                    family="device",
                    claim="This device profile has been used on the card before",
                    ref="query:card_baseline(card_id)",
                    entity_ids=(flagged.txn_id,),
                )
            )
    if baseline is None or baseline.n_txns < c.BASELINE_MIN_TXNS:
        signals.append(
            Signal(
                name="no_baseline",
                family="behaviour",
                claim="The card has too little history to judge what is unusual for it",
                ref="query:card_baseline(card_id)",
                entity_ids=(evidence.card_id,),
            )
        )
    return signals


def _trigger_signal(trigger: Trigger, evidence: CaseEvidence) -> Signal | None:
    if trigger is Trigger.CUSTOMER_REPORT:
        return Signal(
            name="customer_dispute",
            family="customer",
            claim="The cardholder reported that they did not make the flagged purchase",
            ref="case_pack:trigger_text",
            entity_ids=(evidence.flagged.txn_id,),
            source=EvidenceSource.CUSTOMER,
        )
    if trigger is Trigger.ANALYST_REQUEST:
        return Signal(
            name="analyst_request",
            family="analyst",
            claim="A fraud analyst asked for this card to be reviewed alongside related activity",
            ref="case_pack:trigger_text",
            entity_ids=(evidence.flagged.txn_id,),
            source=EvidenceSource.EXTERNAL,
        )
    return None


def _similar_case_ids(evidence: CaseEvidence, pattern: Pattern) -> tuple[str, ...]:
    """Prior cases worth citing: same card first, then retrieved look-alikes."""
    chosen: list[str] = [c_.case_id for c_ in evidence.prior_cases if c_.confirmed_fraud][:3]
    for case in evidence.prior_cases:
        if not case.confirmed_fraud and len(chosen) < 3:
            chosen.append(case.case_id)
    for case in evidence.similar_cases:
        if case.case_id not in chosen and (
            pattern is Pattern.NONE or case.pattern == pattern.value
        ):
            chosen.append(case.case_id)
    return tuple(dict.fromkeys(chosen))[:5]


def _prior_pattern_bonus(pattern_counts: dict[str, int], pattern: Pattern) -> float:
    """Case memory nudges towards a pattern this card has suffered before."""
    if not pattern_counts:
        return 0.0
    total = sum(pattern_counts.values())
    return 0.5 * pattern_counts.get(pattern.value, 0) / total


def detect(evidence: CaseEvidence, trigger: Trigger) -> Detection:
    flagged = evidence.flagged
    baseline = evidence.baseline
    window = [t for t in evidence.window if t.ts <= evidence.opened_at]

    signals = _flagged_signals(evidence)
    memory, pattern_counts, confirmed_cases = _memory_signals(evidence)
    signals.extend(memory)
    trigger_signal = _trigger_signal(trigger, evidence)
    if trigger_signal is not None:
        signals.append(trigger_signal)

    testing = card_testing_sequence(window)
    burst = sublimit_burst(window)
    rings = _ring_candidates(evidence)
    region_txns, looks_like_trip = out_of_region_use(evidence, window)
    mixed = mixed_channel_anomaly(window, baseline, flagged)
    cnp = cnp_burst(window, baseline, flagged)
    recurring = recurring_charges(evidence)

    candidates: dict[Pattern, float] = {}
    episodes: dict[Pattern, tuple[Txn, ...]] = {}

    if testing:
        small = [t for t in testing if t.amount <= c.CARD_TESTING_SMALL_MAX_USD]
        cleared = max(
            (t.amount for t in testing if t.amount > c.CARD_TESTING_SMALL_MAX_USD), default=0.0
        )
        signals.append(
            Signal(
                name="card_testing_sequence",
                family="sequence",
                claim=(
                    f"{len(small)} online authorizations under "
                    f"${c.CARD_TESTING_SMALL_MAX_USD:,.0f} within an hour, followed by a "
                    f"${cleared:,.2f} purchase"
                ),
                ref="query:card_window(card_id, hours=2)",
                entity_ids=_ids(testing),
            )
        )
        candidates[Pattern.CARD_TESTING] = 3.0
        episodes[Pattern.CARD_TESTING] = testing

    if burst:
        total = sum(t.amount for t in burst)
        signals.append(
            Signal(
                name="sublimit_burst",
                family="sequence",
                claim=(
                    f"{len(burst)} online purchases within an hour, each between "
                    f"${c.SUBLIMIT_THRESHOLD_USD - c.SUBLIMIT_BAND_USD:,.0f} and "
                    f"${c.SUBLIMIT_THRESHOLD_USD:,.0f}, totalling ${total:,.2f}"
                ),
                ref="query:card_window(card_id, hours=2)",
                entity_ids=_ids(burst),
            )
        )
        candidates[Pattern.UNDOCUMENTED] = 3.0
        episodes[Pattern.UNDOCUMENTED] = burst

    ring_txns: tuple[Txn, ...] = ()
    flagged_rings = [r for r in rings if r.value == flagged.profile_id]
    if flagged_rings:
        ring = flagged_rings[0]
        ring_txns = tuple(t for t in window if t.profile_id == ring.value)
        signals.append(
            Signal(
                name="shared_device_ring",
                family="ring",
                claim=(
                    f"Device profile {ring.value} was used on {ring.card_count} different cards "
                    f"in the surrounding weeks"
                ),
                ref="query:device_reach(profile_id)",
                entity_ids=tuple(ring.cards[:10]),
            )
        )
        if ring.prior_fraud_cases:
            signals.append(
                Signal(
                    name="ring_prior_fraud",
                    family="ring",
                    claim=(
                        f"{len(ring.prior_fraud_cases)} closed case(s) confirmed as fraud involve "
                        f"the same device profile"
                    ),
                    ref="query:device_reach(profile_id)",
                    entity_ids=tuple(ring.prior_fraud_cases[:5]),
                )
            )
        if ring_txns:
            candidates[Pattern.UNDOCUMENTED] = max(candidates.get(Pattern.UNDOCUMENTED, 0.0), 3.5)
            episodes[Pattern.UNDOCUMENTED] = ring_txns

    if region_txns:
        if looks_like_trip:
            signals.append(
                Signal(
                    name="region_trip",
                    family="region",
                    claim=(
                        f"Purchases in region {flagged.region} span several days with no "
                        f"overlapping home-region activity, which reads as a trip"
                    ),
                    ref="query:region_timeline(card_id)",
                    entity_ids=_ids(region_txns),
                )
            )
        else:
            home = baseline.home_region if baseline else ""
            signals.append(
                Signal(
                    name="region_new",
                    family="region",
                    claim=(
                        f"Card-present use in billing region {flagged.region}, where this card has "
                        f"no history, while its home region is {home or 'unknown'}"
                    ),
                    ref="query:region_timeline(card_id)",
                    entity_ids=_ids(region_txns),
                )
            )
            candidates[Pattern.OUT_OF_REGION_USE] = 2.0 + _prior_pattern_bonus(
                pattern_counts, Pattern.OUT_OF_REGION_USE
            )
            episodes[Pattern.OUT_OF_REGION_USE] = region_txns

    if mixed and not looks_like_trip:
        channels = {t.channel for t in mixed}
        signals.append(
            Signal(
                name="mixed_channel",
                family="channel",
                claim=(
                    f"{len(mixed)} transactions across {len(channels)} channels in 48 hours carry "
                    f"device, region or match-flag anomalies"
                ),
                ref="query:card_window(card_id, hours=48)",
                entity_ids=_ids(mixed),
            )
        )
        candidates[Pattern.ACCOUNT_TAKEOVER] = 2.0 + _prior_pattern_bonus(
            pattern_counts, Pattern.ACCOUNT_TAKEOVER
        )
        episodes[Pattern.ACCOUNT_TAKEOVER] = mixed

    if cnp:
        new_device = any(t.from_new_device for t in cnp)
        pattern = (
            Pattern.CARD_NOT_PRESENT_NEW_DEVICE if new_device else Pattern.CARD_NOT_PRESENT_FRAUD
        )
        strength = 2.0 if new_device else 1.0
        if len(cnp) >= c.CNP_BURST_MIN_TXNS:
            signals.append(
                Signal(
                    name="cnp_burst",
                    family="sequence",
                    claim=(
                        f"{len(cnp)} online transactions within 48 hours do not fit the card's "
                        f"history"
                    ),
                    ref="query:card_window(card_id, hours=48)",
                    entity_ids=_ids(cnp),
                )
            )
            strength += 0.5
        candidates[pattern] = strength + _prior_pattern_bonus(pattern_counts, pattern)
        episodes[pattern] = cnp

    if recurring:
        months = len({(t.ts.year, t.ts.month) for t in recurring})
        signals.append(
            Signal(
                name="recurring_charge",
                family="behaviour",
                claim=(
                    f"The same amount (${flagged.amount:,.2f} ± ${c.RECURRING_TOLERANCE_USD:.2f}) "
                    f"and product code appear {len(recurring)} times across {months} months on "
                    f"this card"
                ),
                ref="query:recurring_charges(card_id, amount)",
                entity_ids=_ids(recurring)[:10],
            )
        )

    pattern = max(candidates.items(), key=lambda item: item[1])[0] if candidates else Pattern.NONE

    # Counter-evidence wins where the policy says it should.
    legitimate_explanation = bool(recurring) or looks_like_trip
    if legitimate_explanation and pattern in {
        Pattern.NONE,
        Pattern.OUT_OF_REGION_USE,
        Pattern.CARD_NOT_PRESENT_FRAUD,
    }:
        pattern = Pattern.NONE

    affected = episodes.get(pattern, ())
    if pattern is not Pattern.NONE and flagged.txn_id not in {t.txn_id for t in affected}:
        affected = tuple(sorted([*affected, flagged], key=lambda t: t.ts))

    is_ring = pattern is Pattern.UNDOCUMENTED and bool(ring_txns) and affected == ring_txns
    description = ""
    if pattern is Pattern.UNDOCUMENTED:
        description = _RING_DESCRIPTION if is_ring else _SUBLIMIT_DESCRIPTION

    connected: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    if rings and pattern is not Pattern.NONE:
        connected = tuple(card for card in rings[0].cards if card != evidence.card_id)
        profiles = (rings[0].value,)
    elif pattern is not Pattern.NONE and flagged.profile_id:
        profiles = (flagged.profile_id,)

    conflicts = trigger is Trigger.CUSTOMER_REPORT and pattern is Pattern.NONE
    return Detection(
        pattern=pattern,
        pattern_description=description,
        signals=tuple(signals),
        affected=affected,
        connected_cards=connected,
        device_profiles=profiles,
        recurring_match=bool(recurring),
        shared_origin=bool(connected) and pattern is not Pattern.NONE,
        coordinated_undocumented=is_ring,
        undocumented_pattern=pattern is Pattern.UNDOCUMENTED,
        linked_to_other_fraud=bool(rings and rings[0].prior_fraud_cases),
        evidence_conflicts=conflicts,
        card_testing_cleared_purchase_usd=max(
            (
                t.amount
                for t in episodes.get(Pattern.CARD_TESTING, ())
                if t.amount > c.CARD_TESTING_SMALL_MAX_USD
            ),
            default=0.0,
        ),
        prior_confirmed_cases=tuple(confirmed_cases),
        similar_cases=_similar_case_ids(evidence, pattern),
    )


def memory_query(detection: Detection) -> str:
    """What to search case memory for, in the analysts' vocabulary.

    Built from what the detectors actually found, so retrieval is driven by the evidence
    rather than by the alert's wording.
    """
    parts = [detection.pattern.value.replace("_", " ")]
    parts.extend(signal.claim for signal in detection.signals[:4])
    return " ".join(parts)


def with_similar_cases(detection: Detection, retrieved: tuple[PriorCase, ...]) -> Detection:
    """Add retrieved look-alikes to the cases this investigation cites."""
    chosen = list(detection.similar_cases)
    for case in retrieved:
        if case.case_id in chosen:
            continue
        if detection.pattern is Pattern.NONE or case.pattern == detection.pattern.value:
            chosen.append(case.case_id)
    return replace(detection, similar_cases=tuple(chosen[:5]))
