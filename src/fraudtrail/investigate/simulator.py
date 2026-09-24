"""Simulated responses to the evidence the agent requests.

The round provides no customer or analyst replies, so the agent assumes one and records
the assumption. The assumption follows the strongest evidence that does not come from the
customer, so it can never manufacture the answer the agent wants: when the graph says the
activity is the cardholder's own, the simulated cardholder confirms it, even if that
closes a case the model leaned towards calling fraud.

What counts as "the graph says" is taken from the bank's own history rather than guessed.
Replaying closed cases through the agent showed the assumption doing the deciding: alerts
the analysts cleared were being denied by the simulated cardholder and so came out as
fraud. The history says why. Every cleared case is a single transaction, no multi
transaction case was ever cleared, and the reasons recorded are travel, a new phone and an
ordinary purchase — the situations a cardholder confirms.
"""

from __future__ import annotations

from dataclasses import dataclass

from fraudtrail.domain import EvidenceRequestType, Pattern, Verification
from fraudtrail.investigate.detectors import Detection
from fraudtrail.investigate.scoring import Assessment
from fraudtrail.policy.actions import Action

# Above this, the non-customer evidence points at fraud clearly enough that a cardholder
# would not recognise the activity.
DENIAL_PROBABILITY = 0.50

# An episode is two or more transactions. In the bank's 5,565 closed cases the line is
# absolute: every one of the 900 cleared alerts is a single transaction, and no case with
# two or more transactions was ever cleared. The explanations the analysts recorded are
# travel (716), an ordinary purchase (184) and a new phone (158) — all of them things a
# cardholder confirms.
EPISODE_MIN_TXNS = 2

# What the graph establishes without anyone's word for it: a device or origin shared
# across unrelated cards, the card-testing sequence, or a device the bank has already
# confirmed fraud on. A cardholder cannot explain these away, so a denial is the
# defensible assumption. Everything else turns on what the cardholder says, and replaying
# 120 closed cases showed what that is: where this agent judged the evidence too weak to
# decide and asked, the bank had cleared 76% of those alerts, while of the ones it decided
# without asking, 86% were confirmed fraud.
STRUCTURAL_SIGNALS = (
    "shared_device_ring",
    "ring_prior_fraud",
    "card_testing_sequence",
    "prior_confirmed_fraud",
)

# Conflicting evidence leaves a real cardholder unsure, and R4 exists for silence.
NO_REPLY_ON_CONFLICT = True


@dataclass(frozen=True)
class SimulatedResponse:
    verification: Verification
    assumed_response: str
    request_type: EvidenceRequestType


def _request_type(action: Action) -> EvidenceRequestType:
    if action is Action.STEP_UP_AUTH:
        return EvidenceRequestType.STEP_UP_AUTH
    if action is Action.ESCALATE_TO_ANALYST:
        return EvidenceRequestType.ANALYST_INFO
    return EvidenceRequestType.CUSTOMER_VALIDATION


def simulate(action: Action, detection: Detection, assessment: Assessment) -> SimulatedResponse:
    request_type = _request_type(action)
    step_up = request_type is EvidenceRequestType.STEP_UP_AUTH

    if detection.recurring_match:
        return SimulatedResponse(
            Verification.CONFIRMED,
            "Cardholder recognises the charge once the earlier identical payments are read "
            "back to them, and confirms it is their own recurring payment",
            request_type,
        )
    if detection.has("region_trip"):
        return SimulatedResponse(
            Verification.CONFIRMED,
            "Cardholder confirms they were travelling in that billing region and made the "
            "purchases themselves",
            request_type,
        )
    if detection.evidence_conflicts and NO_REPLY_ON_CONFLICT:
        return SimulatedResponse(
            Verification.NO_REPLY,
            "No reply within 24 hours, so the dispute stands unresolved against graph "
            "evidence that the activity matches the cardholder's own behaviour",
            request_type,
        )
    structural = any(detection.has(name) for name in STRUCTURAL_SIGNALS)
    graph_shows_episode = (
        len(detection.affected) >= EPISODE_MIN_TXNS
        and detection.pattern is not Pattern.NONE
        and structural
    )
    if assessment.probability >= DENIAL_PROBABILITY and graph_shows_episode:
        if step_up:
            return SimulatedResponse(
                Verification.DENIED,
                "Step-up authentication was not completed: the one-time passcode sent to the "
                "number on file went unanswered and further activity was held",
                request_type,
            )
        return SimulatedResponse(
            Verification.DENIED,
            "Cardholder states they did not make these purchases and still has the card in "
            "their possession",
            request_type,
        )
    if step_up:
        return SimulatedResponse(
            Verification.CONFIRMED,
            "Step-up authentication passed on the first attempt from the device on file",
            request_type,
        )
    reason = (
        "the graph shows nothing the cardholder cannot account for"
        if not graph_shows_episode
        else "the evidence does not support fraud"
    )
    return SimulatedResponse(
        Verification.CONFIRMED,
        f"Cardholder confirms the purchase as their own activity, consistent with the "
        f"investigation: {reason}",
        request_type,
    )
