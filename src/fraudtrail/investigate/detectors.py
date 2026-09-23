"""Pattern detection over retrieved evidence.

Each detector turns evidence into named signals and, when it matches, the set of
transactions that belong to the episode. Nothing here decides an action or a probability:
`scoring.py` weighs the signals and `policy/engine.py` decides. Detectors state what they
saw and which query showed it, so every claim in an answer file can be traced.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

from fraudtrail.domain import EvidenceSource, Pattern
from fraudtrail.evidence.models import CardBaseline, CaseEvidence, Txn
from fraudtrail.investigate import constants as c


@dataclass(frozen=True)
class Signal:
    """One thing the evidence shows, with where it came from."""

    name: str
    family: str
    """Signals in the same family are not independent evidence of each other."""

    claim: str
    ref: str
    entity_ids: tuple[str, ...] = ()
    source: EvidenceSource = EvidenceSource.GRAPH


@dataclass(frozen=True)
class Detection:
    pattern: Pattern
    pattern_description: str
    signals: tuple[Signal, ...]
    affected: tuple[Txn, ...]
    connected_cards: tuple[str, ...] = ()
    device_profiles: tuple[str, ...] = ()
    recurring_match: bool = False
    shared_origin: bool = False
    coordinated_undocumented: bool = False
    undocumented_pattern: bool = False
    linked_to_other_fraud: bool = False
    evidence_conflicts: bool = False
    card_testing_cleared_purchase_usd: float = 0.0
    prior_confirmed_cases: tuple[str, ...] = ()
    similar_cases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def exposure_usd(self) -> float:
        return round(sum(abs(t.amount) for t in self.affected), 2)

    @property
    def affected_ids(self) -> tuple[str, ...]:
        return tuple(t.txn_id for t in self.affected)

    @property
    def first_suspicious_txn_id(self) -> str:
        return self.affected[0].txn_id if self.affected else ""

    @property
    def signal_names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.signals)

    @property
    def families(self) -> frozenset[str]:
        return frozenset(s.family for s in self.signals)

    def has(self, name: str) -> bool:
        return name in self.signal_names


def _sorted(txns: Iterable[Txn]) -> tuple[Txn, ...]:
    return tuple(sorted(txns, key=lambda t: t.ts))


def _amount_is_unusual(baseline: CardBaseline | None, txn: Txn) -> bool:
    if baseline is None or baseline.n_txns < c.BASELINE_MIN_TXNS:
        return False
    return baseline.amount_z(txn.amount) >= c.AMOUNT_UNUSUAL_Z


def _product_is_new(baseline: CardBaseline | None, txn: Txn) -> bool:
    if baseline is None or baseline.n_txns < c.BASELINE_MIN_TXNS:
        return False
    return not baseline.product_seen(txn.product_cd)


def recurring_charges(evidence: CaseEvidence) -> tuple[Txn, ...]:
    """Earlier charges that look like the flagged one repeating (R7)."""
    flagged = evidence.flagged
    matches = [
        t
        for t in evidence.recurring_candidates
        if t.txn_id != flagged.txn_id
        and t.product_cd == flagged.product_cd
        and abs(t.amount - flagged.amount) <= c.RECURRING_TOLERANCE_USD
    ]
    months = {(t.ts.year, t.ts.month) for t in matches}
    if len(matches) >= c.RECURRING_MIN_OCCURRENCES and len(months) >= c.RECURRING_MIN_MONTHS:
        return _sorted(matches)
    return ()


def card_testing_sequence(window: Sequence[Txn]) -> tuple[Txn, ...]:
    """Three or more small online authorizations within an hour, then a larger purchase."""
    online = [t for t in window if t.is_online]
    for i, start in enumerate(online):
        if start.amount > c.CARD_TESTING_SMALL_MAX_USD:
            continue
        cutoff = start.ts + timedelta(minutes=c.CARD_TESTING_WINDOW_MINUTES)
        small = [
            t for t in online[i:] if t.ts <= cutoff and t.amount <= c.CARD_TESTING_SMALL_MAX_USD
        ]
        if len(small) < c.CARD_TESTING_MIN_SMALL_AUTHS:
            continue
        after = small[-1].ts
        larger = [
            t
            for t in online
            if t.ts >= after
            and t.amount > c.CARD_TESTING_SMALL_MAX_USD
            and t.ts <= after + timedelta(hours=c.CNP_BURST_HOURS)
        ]
        if larger:
            return _sorted([*small, *larger])
    return ()


def sublimit_burst(window: Sequence[Txn]) -> tuple[Txn, ...]:
    """Several online purchases inside an hour, each just under the $500 limit.

    The shape the bank's analysts recorded as undocumented in CC-3748 and four others.
    """
    band_low = c.SUBLIMIT_THRESHOLD_USD - c.SUBLIMIT_BAND_USD
    candidates = [
        t for t in window if t.is_online and band_low <= t.amount < c.SUBLIMIT_THRESHOLD_USD
    ]
    for i, start in enumerate(candidates):
        cutoff = start.ts + timedelta(minutes=c.SUBLIMIT_WINDOW_MINUTES)
        burst = [t for t in candidates[i:] if t.ts <= cutoff]
        if len(burst) >= c.SUBLIMIT_MIN_TXNS:
            return _sorted(burst)
    return ()


def cnp_burst(
    window: Sequence[Txn], baseline: CardBaseline | None, flagged: Txn
) -> tuple[Txn, ...]:
    """Online transactions around the flagged one that do not fit the card's history."""
    if not flagged.is_online:
        return ()
    start = flagged.ts - timedelta(hours=c.CNP_BURST_HOURS)
    end = flagged.ts + timedelta(hours=c.CNP_BURST_HOURS)
    burst = [
        t
        for t in window
        if t.is_online
        and start <= t.ts <= end
        and (
            t.txn_id == flagged.txn_id
            # The same device is the strongest reason to treat another purchase as part
            # of this episode; novelty alone sweeps in ordinary shopping.
            or (bool(t.profile_id) and t.profile_id == flagged.profile_id)
            or _amount_is_unusual(baseline, t)
            or _product_is_new(baseline, t)
        )
    ]
    return _sorted(burst)


def out_of_region_use(
    evidence: CaseEvidence, window: Sequence[Txn]
) -> tuple[tuple[Txn, ...], bool]:
    """In-person use in a region with no history, and whether it looks like a trip.

    A trip shows several days in the new region. A clone shows home-region activity
    continuing at the same time.
    """
    flagged = evidence.flagged
    baseline = evidence.baseline
    if flagged.is_online or not flagged.region:
        return (), False
    if baseline is None or baseline.region_seen(flagged.region):
        return (), False

    in_region = _sorted(t for t in window if t.region == flagged.region and not t.is_online)
    if not in_region:
        in_region = (flagged,)
    span_days = (in_region[-1].ts - in_region[0].ts).total_seconds() / 86_400
    home = baseline.home_region
    home_nearby = [
        t
        for t in window
        if t.region == home
        and home
        and abs((t.ts - flagged.ts).total_seconds()) <= c.CLONE_HOME_ACTIVITY_HOURS * 3600
    ]
    looks_like_trip = span_days >= c.TRIP_MIN_DAYS and not home_nearby
    return in_region, looks_like_trip


def mixed_channel_anomaly(
    window: Sequence[Txn], baseline: CardBaseline | None, flagged: Txn
) -> tuple[Txn, ...]:
    """Activity on both channels that does not fit the cardholder (account takeover)."""
    start = flagged.ts - timedelta(hours=c.CNP_BURST_HOURS)
    suspicious = [
        t
        for t in window
        if start <= t.ts <= flagged.ts
        and (
            t.from_new_device
            or t.behind_proxy
            or _amount_is_unusual(baseline, t)
            or (baseline is not None and t.region and not baseline.region_seen(t.region))
            or c.MATCH_FLAG_FALSE in _match_flag_values(t)
        )
    ]
    channels = {t.channel for t in suspicious}
    if len(channels) >= c.ATO_MIN_CHANNELS:
        return _sorted(suspicious)
    return ()


def _match_flag_values(txn: Txn) -> set[str]:
    values: set[str] = set()
    for part in txn.m_flags.split("|"):
        if ":" in part:
            values.add(part.split(":", 1)[1])
    return values
