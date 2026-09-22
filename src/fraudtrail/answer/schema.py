"""Pydantic model of one answer file, field for field as the README's Answer Format defines it.

Field order matches the README so serialized files read in the documented order.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from fraudtrail.domain import (
    CaseStatus,
    EvidenceRequestType,
    EvidenceSource,
    Pattern,
    Verdict,
)
from fraudtrail.policy.actions import Action, Route

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Usd = Annotated[float, Field(ge=0.0)]
NonEmpty = Annotated[str, Field(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(_Strict):
    claim: NonEmpty
    source: EvidenceSource
    ref: NonEmpty
    entity_ids: list[str]


class Case(_Strict):
    status: CaseStatus
    verdict: Verdict
    fraud_probability: Probability
    pattern: Pattern
    pattern_description: str
    affected_txn_ids: list[str]
    first_suspicious_txn_id: str
    connected_card_ids: list[str]
    connected_device_profiles: list[str]
    exposure_usd: Usd
    evidence: list[Evidence]
    similar_prior_cases: list[str]
    summary: NonEmpty
    written_to_graph: bool
    graph_case_id: str


class EvidenceRequest(_Strict):
    type: EvidenceRequestType
    asked_after_step: Annotated[int, Field(ge=1)]
    assumed_response: NonEmpty


class ActionItem(_Strict):
    action: Action
    route: Route
    reason: NonEmpty


class NextBestActions(_Strict):
    initial: Annotated[list[ActionItem], Field(min_length=1)]
    final: Annotated[list[ActionItem], Field(min_length=1)]
    what_changed: NonEmpty


class Sar(_Strict):
    file: bool
    reason: NonEmpty
    narrative: str
    subjects: list[str]
    total_amount_usd: Usd
    activity_dates: list[str]


class Answer(_Strict):
    case_id: Annotated[str, Field(pattern=r"^HHG-\d{3}$")]
    case: Case
    evidence_requests: list[EvidenceRequest]
    next_best_actions: NextBestActions
    sar: Sar
    stop_reason: NonEmpty
    tool_calls: Annotated[int, Field(ge=0)]
    tokens: Annotated[int, Field(ge=0)]
    latency_s: Annotated[float, Field(ge=0.0)]
