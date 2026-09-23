"""Fraud probability from the detected signals.

An additive log-odds model: each signal moves the odds by a fixed amount, and a
verification response moves them again. Weights start from what the closed cases show
(docs/data_findings.md) and are checked by the evaluation harness, which measures
calibration against 5,565 decided cases rather than tuning to an answer key.

The probability is honest about uncertainty on purpose: the policy's thresholds (0.30,
0.70, 0.85, 0.15) only behave sensibly if the number means what it says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fraudtrail.domain import Verification
from fraudtrail.investigate.detectors import Detection

# Half the exam cases are legitimate, so an alert starts at even odds.
PRIOR_LOG_ODDS = 0.0

# How much each signal moves the log odds. Positive means more likely fraud.
SIGNAL_WEIGHTS: dict[str, float] = {
    # Sequences the policy names as patterns in their own right.
    "card_testing_sequence": 2.5,
    "sublimit_burst": 2.5,
    "shared_device_ring": 2.0,
    "ring_prior_fraud": 1.0,
    "cnp_burst": 0.8,
    # Card-present use somewhere the card has never been, with home activity continuing.
    "region_new": 1.2,
    "mixed_channel": 1.0,
    # Device and connection.
    "new_device": 0.6,
    "proxy": 0.8,
    "device_seen_before": -0.7,
    # Behaviour against the card's own history.
    "amount_unusual": 0.5,
    "product_new": 0.4,
    # The cardholder and the analysts.
    "customer_dispute": 1.8,
    "analyst_request": 0.5,
    # Memory.
    "prior_confirmed_fraud": 0.5,
    "prior_cleared_travel": -0.6,
    "prior_cleared_new_phone": -0.5,
    # Counter-evidence.
    "region_trip": -1.8,
    "recurring_charge": -2.2,
    # The model's own score is weak evidence here: above 0.70, most flagged transactions
    # in this book are legitimate, and confirmed fraud appears in every score band.
    "model_score_high": -0.4,
    # No history to judge against, so novelty signals mean less.
    "no_baseline": -0.2,
}

# What a verification response is worth once it arrives.
RESPONSE_WEIGHTS: dict[Verification, float] = {
    Verification.DENIED: 2.5,
    Verification.CONFIRMED: -3.0,
    Verification.NO_REPLY: 0.2,
}

# Section 6: "A verification response settles the question." A cardholder who recognises
# the activity ends the case even when other signals looked bad, and a denial confirms it.
CONFIRMED_MAX_PROBABILITY = 0.10
DENIED_MIN_PROBABILITY = 0.80

# Probabilities never reach certainty: something is always unobserved.
MIN_PROBABILITY = 0.02
MAX_PROBABILITY = 0.97

# A signal family counts as independent evidence when it moves the odds at least this far.
INDEPENDENT_EVIDENCE_MIN_WEIGHT = 0.5

# Families that describe the alert rather than the behaviour behind it.
NON_EVIDENCE_FAMILIES = frozenset({"model"})


@dataclass(frozen=True)
class Contribution:
    name: str
    family: str
    weight: float


@dataclass(frozen=True)
class Assessment:
    probability: float
    independent_evidence: int
    contributions: tuple[Contribution, ...]

    @property
    def top_drivers(self) -> tuple[str, ...]:
        ranked = sorted(self.contributions, key=lambda c: abs(c.weight), reverse=True)
        return tuple(c.name for c in ranked[:3])


def _sigmoid(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-log_odds))


def assess(detection: Detection, verification: Verification | None = None) -> Assessment:
    contributions: list[Contribution] = []
    seen: set[str] = set()
    for signal in detection.signals:
        if signal.name in seen:
            continue
        seen.add(signal.name)
        weight = SIGNAL_WEIGHTS.get(signal.name)
        if weight is None:
            continue
        contributions.append(Contribution(signal.name, signal.family, weight))

    if verification is not None:
        contributions.append(
            Contribution(
                f"verification_{verification.value}", "customer", RESPONSE_WEIGHTS[verification]
            )
        )

    log_odds = PRIOR_LOG_ODDS + sum(c.weight for c in contributions)
    probability = min(max(_sigmoid(log_odds), MIN_PROBABILITY), MAX_PROBABILITY)
    if verification is Verification.CONFIRMED:
        probability = min(probability, CONFIRMED_MAX_PROBABILITY)
    elif verification is Verification.DENIED:
        probability = max(probability, DENIED_MIN_PROBABILITY)

    families = {
        c.family
        for c in contributions
        if abs(c.weight) >= INDEPENDENT_EVIDENCE_MIN_WEIGHT
        and c.family not in NON_EVIDENCE_FAMILIES
    }
    return Assessment(
        probability=round(probability, 2),
        independent_evidence=len(families),
        contributions=tuple(contributions),
    )
