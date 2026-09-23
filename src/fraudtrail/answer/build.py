"""Assemble the answer file.

Code fills every field an answer is scored on: IDs, exposure, routes, the report flag,
dates and counters. The narrator only supplies prose. That split is what keeps invented
IDs and arithmetic mistakes out of the submission.
"""

from __future__ import annotations

from dataclasses import dataclass

from fraudtrail.answer.narration import Narration, Narrator
from fraudtrail.answer.schema import (
    ActionItem,
    Answer,
    Case,
    Evidence,
    EvidenceRequest,
    NextBestActions,
    Sar,
)
from fraudtrail.casepack import ExamCase
from fraudtrail.domain import CaseStatus, Pattern, Verdict, Verification
from fraudtrail.evidence.models import CaseEvidence
from fraudtrail.investigate.detectors import Detection
from fraudtrail.investigate.gather import ToolCall
from fraudtrail.investigate.scoring import Assessment
from fraudtrail.investigate.simulator import SimulatedResponse
from fraudtrail.policy.engine import Decision, verdict_for

# The evidence list carries the detail; more than this is noise in a case file.
MAX_EVIDENCE_ITEMS = 8


@dataclass(frozen=True)
class RunCost:
    tool_calls: int
    tokens: int
    latency_s: float


@dataclass(frozen=True)
class InvestigationResult:
    """Everything one investigation produced, before it becomes JSON."""

    case: ExamCase
    evidence: CaseEvidence
    detection: Detection
    initial_assessment: Assessment
    final_assessment: Assessment
    initial: Decision
    final: Decision
    response: SimulatedResponse | None
    asked_after_step: int
    calls: tuple[ToolCall, ...]
    """Every evidence query this investigation made, in order, for the case event trail."""

    graph_case_id: str
    written_to_graph: bool
    cost: RunCost

    @property
    def verification(self) -> Verification | None:
        return self.response.verification if self.response else None

    @property
    def verdict(self) -> Verdict:
        return verdict_for(self.final_assessment.probability)


def _actions(decision: Decision) -> list[ActionItem]:
    return [ActionItem(action=r.action, route=r.route, reason=r.reason) for r in decision.actions]


def _what_changed(result: InvestigationResult) -> str:
    if result.response is None:
        return "nothing"
    before = result.initial_assessment.probability
    after = result.final_assessment.probability
    added = [r.action.value for r in result.final.actions]
    removed = [
        r.action.value
        for r in result.initial.actions
        if r.action not in {a.action for a in result.final.actions}
    ]
    changes = f"probability moved from {before:.2f} to {after:.2f}"
    if added:
        changes += f"; the recommendation is now {', '.join(added)}"
    if removed:
        changes += f"; {', '.join(removed)} no longer applies"
    return f"The assumed response ({result.response.verification.value}) {changes}."


def _evidence_items(result: InvestigationResult) -> list[Evidence]:
    items = [
        Evidence(
            claim=signal.claim,
            source=signal.source,
            ref=signal.ref,
            entity_ids=list(signal.entity_ids),
        )
        for signal in result.detection.signals[:MAX_EVIDENCE_ITEMS]
    ]
    if result.response is not None:
        items.append(
            Evidence(
                claim=result.response.assumed_response,
                source="customer",  # type: ignore[arg-type]
                ref="evidence_request:1",
                entity_ids=[result.case.flagged_txn_id],
            )
        )
    return items


def _subjects(result: InvestigationResult) -> list[str]:
    subjects = [result.case.customer_id, result.case.card_id]
    subjects.extend(result.detection.connected_cards[:10])
    subjects.extend(result.detection.device_profiles[:3])
    return list(dict.fromkeys(subjects))


def build_answer(result: InvestigationResult, narrator: Narrator) -> Answer:
    detection = result.detection
    verdict = result.verdict
    legitimate = verdict is Verdict.LEGITIMATE

    pattern = Pattern.NONE if legitimate else detection.pattern
    affected = [] if legitimate else list(detection.affected_ids)
    exposure = 0.0 if legitimate else detection.exposure_usd
    first_suspicious = "" if legitimate else detection.first_suspicious_txn_id

    narration = Narration(
        case=result.case,
        evidence=result.evidence,
        detection=detection,
        assessment=result.final_assessment,
        verdict=verdict,
        decision=result.final,
        verification=result.verification,
    )

    status = result.final.status
    if legitimate and status is CaseStatus.CLOSED_FRAUD:
        status = CaseStatus.CLOSED_LEGITIMATE

    case = Case(
        status=status,
        verdict=verdict,
        fraud_probability=result.final_assessment.probability,
        pattern=pattern,
        pattern_description=detection.pattern_description
        if pattern is Pattern.UNDOCUMENTED
        else "",
        affected_txn_ids=affected,
        first_suspicious_txn_id=first_suspicious,
        connected_card_ids=[] if legitimate else list(detection.connected_cards),
        connected_device_profiles=[] if legitimate else list(detection.device_profiles),
        exposure_usd=exposure,
        evidence=_evidence_items(result),
        similar_prior_cases=list(detection.similar_cases),
        summary=narrator.summary(narration),
        written_to_graph=result.written_to_graph,
        graph_case_id=result.graph_case_id if result.written_to_graph else "",
    )

    requests: list[EvidenceRequest] = []
    if result.response is not None:
        requests.append(
            EvidenceRequest(
                type=result.response.request_type,
                asked_after_step=result.asked_after_step,
                assumed_response=result.response.assumed_response,
            )
        )

    file_report = result.final.report.file and not legitimate
    first_date, last_date = narration.dates
    sar = Sar(
        file=file_report,
        reason=result.final.report.reason,
        narrative=narrator.sar_narrative(narration) if file_report else "",
        subjects=_subjects(result) if file_report else [],
        total_amount_usd=exposure if file_report else 0.0,
        activity_dates=[first_date, last_date] if file_report else [],
    )

    return Answer(
        case_id=result.case.case_id,
        case=case,
        evidence_requests=requests,
        next_best_actions=NextBestActions(
            initial=_actions(result.initial),
            final=_actions(result.final),
            what_changed=_what_changed(result),
        ),
        sar=sar,
        stop_reason=result.final.stop_reason,
        tool_calls=result.cost.tool_calls,
        tokens=result.cost.tokens,
        latency_s=round(result.cost.latency_s, 2),
    )
