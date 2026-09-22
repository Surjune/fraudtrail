"""Enumerations shared by the policy engine, the answer schema and the agent.

Values are the exact identifiers the dataset README's Answer Format requires.
"""

from __future__ import annotations

from enum import StrEnum


class Trigger(StrEnum):
    RISK_SCORE = "risk_score"
    CUSTOMER_REPORT = "customer_report"
    ANALYST_REQUEST = "analyst_request"


class Verdict(StrEnum):
    FRAUD = "fraud"
    LEGITIMATE = "legitimate"
    UNCERTAIN = "uncertain"


class CaseStatus(StrEnum):
    OPEN = "open"
    CLOSED_FRAUD = "closed_fraud"
    CLOSED_LEGITIMATE = "closed_legitimate"
    ESCALATED = "escalated"


class Pattern(StrEnum):
    CARD_TESTING = "card_testing"
    CARD_NOT_PRESENT_FRAUD = "card_not_present_fraud"
    CARD_NOT_PRESENT_NEW_DEVICE = "card_not_present_new_device"
    OUT_OF_REGION_USE = "out_of_region_use"
    ACCOUNT_TAKEOVER = "account_takeover"
    UNDOCUMENTED = "undocumented"
    NONE = "none"


class EvidenceSource(StrEnum):
    GRAPH = "graph"
    DOCUMENT = "document"
    CUSTOMER = "customer"
    EXTERNAL = "external"


class EvidenceRequestType(StrEnum):
    CUSTOMER_VALIDATION = "customer_validation"
    STEP_UP_AUTH = "step_up_auth"
    ANALYST_INFO = "analyst_info"


class Verification(StrEnum):
    """The cardholder's answer to a verification or step-up request."""

    CONFIRMED = "confirmed"
    """Made the transaction, or passed step-up authentication."""

    DENIED = "denied"
    """Did not make the transaction, or failed step-up authentication."""

    NO_REPLY = "no_reply"
    """No answer within the R4 window."""
