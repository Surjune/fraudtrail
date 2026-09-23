"""Simulated responses to the evidence the agent requests.

The round provides no customer or analyst replies, so the agent assumes one and records
the assumption. The assumption follows the strongest evidence that does not come from the
customer, so it can never manufacture the answer the agent wants: when the graph says the
activity is the cardholder's own, the simulated cardholder confirms it, even if that
closes a case the model leaned towards calling fraud.
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
    graph_shows_episode = bool(detection.affected) and detection.pattern is not Pattern.NONE
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
        "the graph shows no episode behind the alert"
        if not graph_shows_episode
        else "the evidence does not support fraud"
    )
    return SimulatedResponse(
        Verification.CONFIRMED,
        f"Cardholder confirms the purchase as their own activity, consistent with the "
        f"investigation: {reason}",
        request_type,
    )
