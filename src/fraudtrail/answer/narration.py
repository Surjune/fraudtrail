"""The written parts of an answer: the case summary and the report narrative.

`TemplateNarrator` builds both from the evidence alone, so the agent produces complete,
defensible answer files with no model configured. When an LLM is configured it replaces
the wording only, never the facts, the IDs or the decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fraudtrail.casepack import ExamCase
from fraudtrail.domain import Pattern, Verdict, Verification
from fraudtrail.evidence.models import CaseEvidence
from fraudtrail.investigate.detectors import Detection
from fraudtrail.investigate.scoring import Assessment
from fraudtrail.policy.engine import Decision
from fraudtrail.wording import counted, noun, was_or_were

# The summary is for an analyst: the format asks for two to six sentences.
MAX_SUMMARY_CLAIMS = 3

# The narrative is for a regulator: six to twelve sentences, standing on its own.
MAX_NARRATIVE_CLAIMS = 4


@dataclass(frozen=True)
class Narration:
    case: ExamCase
    evidence: CaseEvidence
    detection: Detection
    assessment: Assessment
    verdict: Verdict
    decision: Decision
    verification: Verification | None

    @property
    def dates(self) -> tuple[str, str]:
        affected = self.detection.affected
        if not affected:
            flagged = self.evidence.flagged.ts.date().isoformat()
            return flagged, flagged
        return affected[0].ts.date().isoformat(), affected[-1].ts.date().isoformat()


class Narrator(Protocol):
    def summary(self, n: Narration) -> str: ...

    def sar_narrative(self, n: Narration) -> str: ...


def _trigger_sentence(n: Narration) -> str:
    flagged = n.evidence.flagged
    where = f"in billing region {flagged.region}" if flagged.region else "online"
    what = f"a ${flagged.amount:,.2f} {flagged.channel.replace('_', '-')} purchase {where}"
    if n.case.risk_score is not None:
        return f"The bank's model scored {what} on card {n.case.card_id} at {n.case.risk_score:.2f}"
    if n.case.trigger.value == "customer_report":
        return f"Customer {n.case.customer_id} disputed {what} on card {n.case.card_id}"
    return f"An analyst asked for a review of {what} on card {n.case.card_id}"


def _claims(n: Narration, limit: int) -> list[str]:
    ranked = [s for s in n.detection.signals if s.name != "no_baseline"]
    return [s.claim for s in ranked[:limit]]


def _response_sentence(n: Narration) -> str:
    if n.verification is Verification.DENIED:
        return "The cardholder was asked to validate the activity and denied it"
    if n.verification is Verification.CONFIRMED:
        return "The cardholder was asked to validate the activity and confirmed it as their own"
    if n.verification is Verification.NO_REPLY:
        return "The cardholder was asked to validate the activity and did not reply within 24 hours"
    return ""


def _outcome_sentence(n: Narration) -> str:
    actions = ", ".join(r.action.value for r in n.decision.actions)
    return (
        f"Assessed as {n.verdict.value} at {n.assessment.probability:.2f}, with "
        f"{n.detection.exposure_usd:,.2f} USD exposure; recommended actions are {actions}"
    )


class TemplateNarrator:
    """Writes both texts from the evidence, with no model involved."""

    def summary(self, n: Narration) -> str:
        parts = [_trigger_sentence(n)]
        parts.extend(_claims(n, MAX_SUMMARY_CLAIMS))
        response = _response_sentence(n)
        if response:
            parts.append(response)
        parts.append(_outcome_sentence(n))
        return ". ".join(part.rstrip(".") for part in parts if part) + "."

    def sar_narrative(self, n: Narration) -> str:
        flagged = n.evidence.flagged
        affected = n.detection.affected
        first, last = n.dates
        channels = sorted({t.channel.replace("_", "-") for t in affected}) or [
            flagged.channel.replace("_", "-")
        ]
        products = sorted({t.product_cd for t in affected}) or [flagged.product_cd]
        regions = sorted({t.region for t in affected if t.region})
        where = (
            f"{noun(len(regions), 'billing region')} {', '.join(regions)}"
            if regions
            else "no recorded billing region"
        )

        sentences = [
            f"Between {first} and {last}, card {n.case.card_id} belonging to customer "
            f"{n.case.customer_id} was used for {counted(len(affected), 'transaction')} "
            f"totalling ${n.detection.exposure_usd:,.2f} that the bank believes "
            f"{was_or_were(len(affected))} unauthorised",
            f"The activity ran through the {', '.join(channels)} "
            f"{noun(len(channels), 'channel')} under {noun(len(products), 'product code')} "
            f"{', '.join(products)}, in {where}",
        ]
        sentences.extend(_claims(n, MAX_NARRATIVE_CLAIMS))
        if n.detection.pattern is Pattern.UNDOCUMENTED and n.detection.pattern_description:
            sentences.append(n.detection.pattern_description.rstrip("."))
        if n.detection.connected_cards:
            sentences.append(
                f"The same device profile links this activity to "
                f"{counted(len(n.detection.connected_cards), 'other card')}, which indicates "
                f"one actor operating across several cardholders"
            )
        response = _response_sentence(n)
        if response:
            sentences.append(response)
        sentences.append(
            f"The total suspicious amount is ${n.detection.exposure_usd:,.2f}, and the "
            f"recommended actions are "
            f"{', '.join(r.action.value for r in n.decision.actions)}"
        )
        # The format allows six to twelve sentences; keep the strongest ones.
        return ". ".join(s.rstrip(".") for s in sentences[:12] if s) + "."
