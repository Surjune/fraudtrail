"""One investigation, end to end.

Trigger, gather, assess, decide, request evidence if the policy calls for one, reassess,
decide again, explain. The deterministic pieces make every decision; the narrator writes
the prose; the graph stores the result as memory for the next case.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter

from fraudtrail.answer.build import InvestigationResult, RunCost
from fraudtrail.casepack import ExamCase
from fraudtrail.domain import Pattern, Verification
from fraudtrail.evidence.provider import EvidenceProvider
from fraudtrail.investigate.analysis import detect, memory_query, with_similar_cases
from fraudtrail.investigate.detectors import Detection
from fraudtrail.investigate.gather import Gathered, gather
from fraudtrail.investigate.scoring import Assessment, assess
from fraudtrail.investigate.simulator import simulate
from fraudtrail.policy.engine import Decision, Situation, decide

log = logging.getLogger(__name__)

CASE_ID_PREFIX = "CASE"

MemoryWriter = Callable[[ExamCase, InvestigationResult], bool]
"""Stores a finished investigation in the graph; returns whether it was written."""


def graph_case_id(case: ExamCase) -> str:
    return f"{CASE_ID_PREFIX}-{case.case_id}"


def _cards_with_confirmed_fraud(gathered: Gathered) -> int:
    return len({case.card_id for case in gathered.evidence.prior_cases if case.confirmed_fraud})


def build_situation(
    case: ExamCase,
    detection: Detection,
    assessment: Assessment,
    gathered: Gathered,
    verification: Verification | None,
    case_opened: bool,
) -> Situation:
    return Situation(
        trigger=case.trigger,
        fraud_probability=assessment.probability,
        independent_evidence=assessment.independent_evidence,
        exposure_usd=detection.exposure_usd,
        flagged_is_pending=case.flagged_is_pending,
        connected_cards=len(detection.connected_cards),
        card_testing=detection.pattern is Pattern.CARD_TESTING,
        card_testing_cleared_purchase_usd=detection.card_testing_cleared_purchase_usd,
        recurring_charge_match=detection.recurring_match,
        shared_origin=detection.shared_origin,
        linked_to_other_fraud=detection.linked_to_other_fraud,
        coordinated_undocumented=detection.coordinated_undocumented,
        undocumented_pattern=detection.undocumented_pattern,
        evidence_conflicts=detection.evidence_conflicts,
        cards_with_confirmed_fraud=_cards_with_confirmed_fraud(gathered),
        verification=verification,
        case_opened=case_opened,
    )


def investigate(
    provider: EvidenceProvider,
    case: ExamCase,
    memory_writer: MemoryWriter | None = None,
    tokens_used: int = 0,
) -> InvestigationResult:
    started = perf_counter()
    gathered = gather(provider, case)
    detection = detect(gathered.evidence, case.trigger)

    # Second pass: retrieve case memory using what the detectors found, not just the
    # alert text. This is the retrieval step the answer's similar_prior_cases rests on.
    retrieved = provider.similar_cases(memory_query(detection), case.opened_at, k=15)
    detection = with_similar_cases(detection, retrieved)
    memory_calls = 1

    initial_assessment = assess(detection)
    initial: Decision = decide(
        build_situation(case, detection, initial_assessment, gathered, None, case_opened=False)
    )

    response = None
    final_assessment = initial_assessment
    final = initial
    if initial.evidence_request is not None:
        response = simulate(initial.evidence_request, detection, initial_assessment)
        final_assessment = assess(detection, response.verification)
        final = decide(
            build_situation(
                case,
                detection,
                final_assessment,
                gathered,
                response.verification,
                case_opened=True,
            )
        )

    result = InvestigationResult(
        case=case,
        evidence=gathered.evidence,
        detection=detection,
        initial_assessment=initial_assessment,
        final_assessment=final_assessment,
        initial=initial,
        final=final,
        response=response,
        asked_after_step=gathered.call_count + memory_calls,
        graph_case_id=graph_case_id(case),
        written_to_graph=False,
        cost=RunCost(
            tool_calls=gathered.call_count + memory_calls,
            tokens=tokens_used,
            latency_s=perf_counter() - started,
        ),
    )

    if memory_writer is not None:
        written = memory_writer(case, result)
        result = InvestigationResult(**{**result.__dict__, "written_to_graph": written})

    log.info(
        "%s: %s p=%.2f exposure=$%.2f actions=%s",
        case.case_id,
        result.detection.pattern.value,
        final_assessment.probability,
        detection.exposure_usd,
        ",".join(r.action.value for r in final.actions),
    )
    return result
