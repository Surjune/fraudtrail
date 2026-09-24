"""Writing a finished investigation back into the graph.

The brief requires the case to be written to the graph, and the reason is not
bookkeeping: a case in the graph is connected to the card, the transactions, the devices
and the closed cases it drew on, so the next investigation can retrieve it the same way
it retrieves the bank's own history. The agent's memory and the bank's memory end up in
one place, in one shape.

Each case carries its own event trail. Every evidence query, the assessment, the decision
before evidence was requested, the response and the decision after it are stored as
`CaseEvent` vertices in order, so an analyst can replay how the case progressed rather
than seeing only where it landed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from fraudtrail.answer.build import InvestigationResult
from fraudtrail.answer.schema import Answer
from fraudtrail.graph.client import GraphError, QueryRunner
from fraudtrail.graph.mcp_client import McpError

log = logging.getLogger(__name__)

TS_FORMAT = "%Y-%m-%d %H:%M:%S"

# Long free text is stored, not the whole world: a detail is one readable line.
MAX_DETAIL = 500

# Edges are written for the cards and devices the case names, capped so one ring does not
# attach hundreds of edges to a single case.
MAX_CONNECTED = 25


@dataclass(frozen=True)
class CaseEvent:
    step: int
    kind: str
    detail: str


def _clip(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:MAX_DETAIL]


def _events(result: InvestigationResult, answer: Answer) -> list[CaseEvent]:
    """The investigation as an ordered trail, one event per thing the agent did."""
    events: list[CaseEvent] = [
        CaseEvent(
            step=1,
            kind="case_opened",
            detail=(
                f"Trigger {result.case.trigger.value} on card {result.case.card_id}, "
                f"transaction {result.case.flagged_txn_id}"
            ),
        )
    ]
    step = 1
    for call in result.calls:
        step += 1
        params = ", ".join(f"{k}={v}" for k, v in call.params.items())
        events.append(
            CaseEvent(
                step=step,
                kind="tool_call",
                detail=_clip(f"{call.name}({params}) returned {call.result_size}"),
            )
        )

    step += 1
    events.append(
        CaseEvent(
            step=step,
            kind="assessment",
            detail=_clip(
                f"Pattern {result.detection.pattern.value} at probability "
                f"{result.initial_assessment.probability:.2f} on "
                f"{result.initial_assessment.independent_evidence} independent signals; "
                f"strongest: {', '.join(result.initial_assessment.top_drivers)}"
            ),
        )
    )
    step += 1
    events.append(
        CaseEvent(
            step=step,
            kind="decision",
            detail=_clip(
                "Before further evidence: "
                + "; ".join(f"{r.action.value} ({r.route.value})" for r in result.initial.actions)
            ),
        )
    )

    if result.response is not None:
        step += 1
        events.append(
            CaseEvent(
                step=step,
                kind="evidence_request",
                detail=_clip(
                    f"Requested {result.response.request_type.value} after step "
                    f"{result.asked_after_step}"
                ),
            )
        )
        step += 1
        events.append(
            CaseEvent(
                step=step,
                kind="evidence_received",
                detail=_clip(
                    f"{result.response.verification.value}: {result.response.assumed_response}"
                ),
            )
        )
        step += 1
        events.append(
            CaseEvent(
                step=step,
                kind="decision",
                detail=_clip(
                    "After the response: "
                    + "; ".join(f"{r.action.value} ({r.route.value})" for r in result.final.actions)
                ),
            )
        )

    step += 1
    events.append(
        CaseEvent(
            step=step,
            kind="report",
            detail=_clip(
                ("Report filed. " if answer.sar.file else "No report. ") + answer.sar.reason
            ),
        )
    )
    step += 1
    events.append(
        CaseEvent(
            step=step,
            kind="case_closed",
            detail=_clip(f"{answer.case.status.value}: {answer.case.summary}"),
        )
    )
    return events


class GraphMemory:
    """Stores finished cases in the graph through the installed write queries."""

    def __init__(self, client: QueryRunner) -> None:
        self._client = client

    def write(self, result: InvestigationResult, answer: Answer) -> bool:
        """Write the case and its event trail. False when the graph refused it."""
        case_id = result.graph_case_id
        try:
            self._client.run_query(
                "write_case",
                {
                    "case_id": case_id,
                    "exam_case_id": answer.case_id,
                    "trigger_type": result.case.trigger.value,
                    "status": answer.case.status.value,
                    "verdict": answer.case.verdict.value,
                    "fraud_probability": answer.case.fraud_probability,
                    "pattern": answer.case.pattern.value,
                    "pattern_description": _clip(answer.case.pattern_description),
                    "exposure_usd": answer.case.exposure_usd,
                    "summary": _clip(answer.case.summary),
                    "report_filed": answer.sar.file,
                    "opened_at": result.case.opened_at.strftime(TS_FORMAT),
                    "card": result.case.card_id,
                    "flagged_txn": result.case.flagged_txn_id,
                    "affected_txns": list(answer.case.affected_txn_ids),
                    "connected_cards": list(answer.case.connected_card_ids[:MAX_CONNECTED]),
                    "device_profiles": list(answer.case.connected_device_profiles[:MAX_CONNECTED]),
                    "prior_cases": list(answer.case.similar_prior_cases[:MAX_CONNECTED]),
                },
            )
            written_at = datetime.now().strftime(TS_FORMAT)
            for event in _events(result, answer):
                self._client.run_query(
                    "write_case_event",
                    {
                        "case_id": case_id,
                        "event_id": f"{case_id}-E{event.step:03d}",
                        "step": event.step,
                        "kind": event.kind,
                        "detail": event.detail,
                        "at": written_at,
                    },
                )
        except (GraphError, McpError) as exc:
            # A case that failed to store is reported as unwritten rather than claimed.
            log.warning("%s was not written to the graph: %s", case_id, exc)
            return False
        return True

    def read(self, case_id: str) -> dict[str, object]:
        """Everything the graph holds about one investigation, for the UI and for checks."""
        blocks = self._client.run_query("read_case", {"case_id": (case_id,)})
        merged: dict[str, object] = {}
        for block in blocks:
            if isinstance(block, dict):
                merged.update(block)
        return merged
