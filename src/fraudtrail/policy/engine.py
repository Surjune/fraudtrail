"""Deterministic Fraud Policy engine.

Turns an assessed situation into the policy's answer: ordered next-best actions with
approval routes, the evidence request (if one is due), whether to file a suspicious
activity report, the case status and why the investigation stops. The LLM never makes
these decisions; it explains them.

A case is decided twice (section 3b): once before any requested evidence comes back
(`initial`) and once after (`final`), each from its own `Situation`.
"""

from __future__ import annotations

from dataclasses import dataclass

from fraudtrail.domain import CaseStatus, Trigger, Verdict, Verification
from fraudtrail.policy import constants as c
from fraudtrail.policy.actions import (
    BLOCKING_ACTIONS,
    EXECUTION_ORDER,
    Action,
    Route,
    route_for,
)


@dataclass(frozen=True)
class Situation:
    """Everything the policy needs to know about a case at one point in time."""

    trigger: Trigger
    fraud_probability: float
    independent_evidence: int
    """Distinct families of evidence behind the assessment (graph behaviour, device,
    region, prior case, customer response...). A risk score alone counts as one."""

    exposure_usd: float
    flagged_is_pending: bool
    """The alert fired on an authorization that can still be declined or allowed."""

    connected_cards: int = 0
    card_testing: bool = False
    card_testing_cleared_purchase_usd: float = 0.0
    """Largest purchase that already cleared after the testing sequence (R5)."""

    recurring_charge_match: bool = False
    """The flagged charge matches the cardholder's own recurring pattern (R7)."""

    shared_origin: bool = False
    """Several cards show fraud from one device profile, region or recipient email (R6)."""

    linked_to_other_fraud: bool = False
    """The case connects to a shared device profile or another card's fraud (R2, 3a)."""

    undocumented_pattern: bool = False
    """The activity fits none of the five documented patterns (3a reporting factor)."""

    coordinated_undocumented: bool = False
    """Activity fits no known pattern and shows coordinated abuse across customers (R9)."""

    evidence_conflicts: bool = False
    cards_with_confirmed_fraud: int = 0
    credentials_compromised: bool = False
    verification: Verification | None = None
    """The cardholder's answer to a request this investigation made."""

    case_opened: bool = False
    """A case already exists from an earlier step of this investigation."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraud_probability <= 1.0:
            raise ValueError(f"fraud_probability out of range: {self.fraud_probability}")
        if self.exposure_usd < 0:
            raise ValueError(f"exposure_usd is negative: {self.exposure_usd}")
        if self.independent_evidence < 0:
            raise ValueError(f"independent_evidence is negative: {self.independent_evidence}")

    @property
    def disputed(self) -> bool:
        return self.trigger is Trigger.CUSTOMER_REPORT

    @property
    def verdict(self) -> Verdict:
        return verdict_for(self.fraud_probability)

    @property
    def single_signal(self) -> bool:
        return self.independent_evidence <= c.R1_SINGLE_SIGNAL_MAX_EVIDENCE


@dataclass(frozen=True)
class Recommendation:
    action: Action
    route: Route
    reason: str


@dataclass(frozen=True)
class ReportDecision:
    file: bool
    reason: str


@dataclass(frozen=True)
class Decision:
    actions: list[Recommendation]
    evidence_request: Action | None
    report: ReportDecision
    status: CaseStatus
    stop_reason: str

    @property
    def action_names(self) -> list[Action]:
        return [r.action for r in self.actions]


def verdict_for(probability: float) -> Verdict:
    if probability >= c.VERDICT_FRAUD_MIN_PROBABILITY:
        return Verdict.FRAUD
    if probability < c.VERDICT_LEGITIMATE_BELOW_PROBABILITY:
        return Verdict.LEGITIMATE
    return Verdict.UNCERTAIN


def stop_reached(s: Situation) -> bool:
    """Section 6, first condition: decisive probability on enough independent evidence."""
    decisive = (
        s.fraud_probability >= c.STOP_FRAUD_MIN_PROBABILITY
        or s.fraud_probability <= c.STOP_LEGITIMATE_MAX_PROBABILITY
    )
    return decisive and s.independent_evidence >= c.STOP_MIN_INDEPENDENT_EVIDENCE


def evidence_request(s: Situation) -> Action | None:
    """The request the policy calls for now, or None when no more evidence is due."""
    if s.verification is not None:
        return None
    if s.disputed:
        if s.recurring_charge_match:
            return Action.VERIFY_WITH_CUSTOMER
        if s.verdict is Verdict.LEGITIMATE:
            return Action.VERIFY_WITH_CUSTOMER
        return None
    if stop_reached(s):
        return None
    if s.card_testing:
        return Action.STEP_UP_AUTH
    return Action.VERIFY_WITH_CUSTOMER


def report_decision(s: Situation, awaiting_evidence: bool) -> ReportDecision:
    """Section 3a: file when fraud is confirmed or strongly suspected and a factor holds."""
    if s.verification is Verification.CONFIRMED:
        return ReportDecision(False, "R3: cardholder confirmed the transaction; no report")
    if s.recurring_charge_match and s.verification is not Verification.DENIED:
        return ReportDecision(False, "R7: disputed charge matches the cardholder's own pattern")
    if s.fraud_probability < c.STRONG_SUSPICION_MIN_PROBABILITY:
        return ReportDecision(
            False,
            f"3a: fraud is neither confirmed nor strongly suspected "
            f"(probability {s.fraud_probability:.2f})",
        )

    mandated: list[str] = []
    if s.shared_origin:
        mandated.append("R6: fraud across several cards shares an origin")
    if s.coordinated_undocumented:
        mandated.append("R9: coordinated abuse fitting no known pattern")
    elif s.undocumented_pattern:
        mandated.append("3a: the pattern is undocumented")
    if s.linked_to_other_fraud:
        mandated.append("3a: activity connects to a shared device profile or another card's fraud")
    if mandated:
        return ReportDecision(True, "; ".join(mandated))

    if s.exposure_usd > c.REPORT_EXPOSURE_THRESHOLD_USD:
        if awaiting_evidence:
            return ReportDecision(
                False,
                "3b: exposure exceeds $1,000 but the filing waits for the requested evidence",
            )
        return ReportDecision(True, f"R2 and 3a: exposure ${s.exposure_usd:,.2f} exceeds $1,000")

    return ReportDecision(
        False,
        "3a: no reporting factor holds (exposure at or under $1,000, no shared origin, "
        "no link to other fraud, not coordinated)",
    )


class _Plan:
    """Collects actions with the rules that justify them, merging repeated additions."""

    def __init__(self) -> None:
        self._reasons: dict[Action, list[str]] = {}

    def add(self, action: Action, reason: str) -> None:
        reasons = self._reasons.setdefault(action, [])
        if reason not in reasons:
            reasons.append(reason)

    def drop(self, action: Action) -> None:
        self._reasons.pop(action, None)

    def __contains__(self, action: object) -> bool:
        return action in self._reasons

    def build(self, exposure_usd: float) -> list[Recommendation]:
        return [
            Recommendation(
                action, route_for(action, exposure_usd), "; ".join(self._reasons[action])
            )
            for action in EXECUTION_ORDER
            if action in self._reasons
        ]


def _block(plan: _Plan, s: Situation, reason: str) -> None:
    all_cards_allowed = (
        s.cards_with_confirmed_fraud >= c.BLOCK_ALL_CARDS_MIN_CONFIRMED_CARDS
        or s.credentials_compromised
    )
    if all_cards_allowed:
        plan.add(
            Action.BLOCK_ALL_CARDS,
            f"{reason}; R10: {s.cards_with_confirmed_fraud} of the customer's cards show confirmed "
            f"fraud or credentials are compromised",
        )
    else:
        plan.add(Action.BLOCK_CARD, reason)
    plan.add(Action.CREATE_CASE, reason)


def _close_legitimate(plan: _Plan, s: Situation, reason: str) -> None:
    if s.flagged_is_pending:
        plan.add(Action.ALLOW_TRANSACTION, reason)
    plan.add(Action.CLOSE_NO_FRAUD, reason)


def _verification_reason(s: Situation, request: Action) -> str:
    if s.card_testing:
        return "R5: testing sequence observed; confirm the cardholder before further activity"
    if s.single_signal and s.fraud_probability < c.R1_SINGLE_SIGNAL_BLOCK_MIN_PROBABILITY:
        return (
            f"R1: probability {s.fraud_probability:.2f} rests on a single signal; "
            f"verify before any block"
        )
    if request is Action.VERIFY_WITH_CUSTOMER:
        return (
            f"5: probability {s.fraud_probability:.2f} is not decisive; ask the cardholder "
            f"before acting"
        )
    return f"5: probability {s.fraud_probability:.2f} is not decisive; request step-up"


def _apply_response(plan: _Plan, s: Situation) -> None:
    if s.verification is Verification.CONFIRMED:
        _close_legitimate(plan, s, "R3: cardholder confirmed the transaction")
        if s.recurring_charge_match and s.disputed:
            plan.add(Action.WARN_CUSTOMER, "R7: remind the cardholder of their recurring charge")
    elif s.verification is Verification.DENIED:
        _block(plan, s, "R2: cardholder denied the transaction")
    elif s.verification is Verification.NO_REPLY:
        plan.add(Action.MONITOR_CARD, "R4: no reply within 24 hours")
        if s.flagged_is_pending:
            plan.add(Action.DECLINE_TRANSACTION, "R4: decline the pending authorization")
        if s.exposure_usd > c.ESCALATION_EXPOSURE_THRESHOLD_USD:
            plan.add(Action.ESCALATE_TO_ANALYST, "R4: no reply and exposure exceeds $500")


def _apply_dispute(plan: _Plan, s: Situation) -> None:
    if s.recurring_charge_match:
        plan.add(Action.CREATE_CASE, "R7 and 3a: customer disputes a charge")
        plan.add(
            Action.VERIFY_WITH_CUSTOMER,
            "R7: disputed charge matches the cardholder's own recurring pattern",
        )
        plan.add(Action.WARN_CUSTOMER, "R7: remind the cardholder of the recurring charge")
    elif s.verdict is Verdict.LEGITIMATE:
        plan.add(Action.CREATE_CASE, "3a: customer disputes a charge")
        plan.add(
            Action.VERIFY_WITH_CUSTOMER,
            "R8: the dispute conflicts with evidence that the activity is the cardholder's own",
        )
    else:
        _block(plan, s, "R2: cardholder reported they did not make the transaction")


def _apply_alert(plan: _Plan, s: Situation, request: Action | None) -> None:
    if s.card_testing:
        if s.flagged_is_pending:
            plan.add(Action.DECLINE_TRANSACTION, "R5: card-testing sequence")
        plan.add(Action.STEP_UP_AUTH, "R5: card-testing sequence")
        if s.card_testing_cleared_purchase_usd > c.CARD_TESTING_CLEARED_PURCHASE_BLOCK_USD:
            plan.add(
                Action.BLOCK_CARD,
                f"R5: a ${s.card_testing_cleared_purchase_usd:,.2f} purchase already cleared "
                f"after testing",
            )

    if stop_reached(s) and s.verdict is Verdict.FRAUD:
        _block(
            plan,
            s,
            f"6: probability {s.fraud_probability:.2f} on {s.independent_evidence} independent "
            f"pieces of evidence",
        )
    elif stop_reached(s) and s.verdict is Verdict.LEGITIMATE:
        _close_legitimate(
            plan,
            s,
            f"6: probability {s.fraud_probability:.2f} on {s.independent_evidence} independent "
            f"pieces of evidence supports closing",
        )
    elif request is not None:
        plan.add(request, _verification_reason(s, request))


def _apply_cross_cutting(plan: _Plan, s: Situation, request: Action | None) -> None:
    confirmed_legitimate = s.verification is Verification.CONFIRMED
    not_cleared = not confirmed_legitimate and s.verdict is not Verdict.LEGITIMATE

    if s.shared_origin and not_cleared:
        plan.add(Action.CREATE_CASE, "R6: shared origin across cards")
        plan.add(Action.MONITOR_CONNECTED_CARDS, "R6: monitor every card sharing the origin")
    if s.coordinated_undocumented and not_cleared:
        plan.add(Action.CREATE_CASE, "R9: coordinated activity fitting no known pattern")
        plan.add(Action.ESCALATE_TO_ANALYST, "R9: coordinated activity fitting no known pattern")

    uncertain_and_exposed = (
        s.verdict is Verdict.UNCERTAIN and s.exposure_usd > c.ESCALATION_EXPOSURE_THRESHOLD_USD
    )
    if not confirmed_legitimate and (uncertain_and_exposed or s.evidence_conflicts):
        why = "evidence conflicts" if s.evidence_conflicts else "uncertain with exposure over $500"
        plan.add(Action.ESCALATE_TO_ANALYST, f"R8: {why}")

    if s.connected_cards > 0 and not_cleared and (_plan_blocks(plan) or s.linked_to_other_fraud):
        plan.add(
            Action.MONITOR_CONNECTED_CARDS,
            f"{s.connected_cards} connected card(s) share the compromise",
        )

    needs_case = (
        s.case_opened
        or s.disputed
        or request is not None
        or s.fraud_probability >= c.CASE_OPEN_MIN_PROBABILITY
    )
    if needs_case:
        plan.add(Action.CREATE_CASE, _case_reason(s, request))


def _plan_blocks(plan: _Plan) -> bool:
    return any(action in plan for action in BLOCKING_ACTIONS)


def _case_reason(s: Situation, request: Action | None) -> str:
    if s.disputed:
        return "3a: customer disputes a charge"
    if request is not None:
        return "3a: evidence requested"
    if s.case_opened:
        return "3a: case opened earlier in this investigation"
    return f"3a: fraud probability {s.fraud_probability:.2f} reaches 0.30"


def _apply_guards(plan: _Plan, s: Situation) -> Action | None:
    """Remove actions the policy forbids here. Returns a verification the guards forced."""
    # R1: never block on a single weak signal.
    weak = s.single_signal and s.fraud_probability < c.R1_SINGLE_SIGNAL_BLOCK_MIN_PROBABILITY
    # R7: never block a disputed charge that matches the cardholder's own pattern.
    recurring = s.recurring_charge_match and s.verification is not Verification.DENIED
    closing = Action.CLOSE_NO_FRAUD in plan
    if weak or recurring or closing:
        for action in BLOCKING_ACTIONS:
            plan.drop(action)
    if closing:
        plan.drop(Action.FILE_REPORT)
        plan.drop(Action.DECLINE_TRANSACTION)
    verifying = Action.VERIFY_WITH_CUSTOMER in plan or Action.STEP_UP_AUTH in plan
    if weak and not verifying and s.verification is None:
        plan.add(Action.VERIFY_WITH_CUSTOMER, "R1: verify before any block on a single signal")
        return Action.VERIFY_WITH_CUSTOMER
    return None


def _status(s: Situation, plan: _Plan, request: Action | None) -> CaseStatus:
    if Action.ESCALATE_TO_ANALYST in plan:
        return CaseStatus.ESCALATED
    if Action.CLOSE_NO_FRAUD in plan:
        return CaseStatus.CLOSED_LEGITIMATE
    if request is None and s.verdict is Verdict.FRAUD and _plan_blocks(plan):
        return CaseStatus.CLOSED_FRAUD
    return CaseStatus.OPEN


def _stop_reason(s: Situation, request: Action | None, status: CaseStatus) -> str:
    if request is not None:
        return f"Paused for evidence: {request.value} requested under section 5"
    if s.verification is Verification.CONFIRMED:
        return "6: the cardholder's confirmation settles the question"
    if s.verification is Verification.DENIED:
        return "6: the cardholder's denial settles the question"
    if s.verification is Verification.NO_REPLY:
        return "R4: no reply within 24 hours; card monitored and the case stays with the bank"
    if stop_reached(s):
        return (
            f"6: probability {s.fraud_probability:.2f} on {s.independent_evidence} independent "
            f"pieces of evidence is decisive"
        )
    if status is CaseStatus.ESCALATED:
        return "6: handed to an analyst; further automated steps would not change the decision"
    return "6: further steps are unlikely to change the decision"


def decide(s: Situation) -> Decision:
    """Apply the Fraud Policy to one situation."""
    plan = _Plan()
    request = evidence_request(s)

    if s.verification is not None:
        _apply_response(plan, s)
    elif s.disputed:
        _apply_dispute(plan, s)
    else:
        _apply_alert(plan, s, request)

    forced = _apply_guards(plan, s)
    if request is None:
        request = forced
    _apply_cross_cutting(plan, s, request)

    report = report_decision(s, awaiting_evidence=request is not None)
    if report.file and Action.CLOSE_NO_FRAUD not in plan:
        plan.add(Action.FILE_REPORT, report.reason)
        plan.add(Action.CREATE_CASE, "3a: a report always has a case behind it")
    else:
        plan.drop(Action.FILE_REPORT)
        report = ReportDecision(False, report.reason)

    status = _status(s, plan, request)
    return Decision(
        actions=plan.build(s.exposure_usd),
        evidence_request=request,
        report=report,
        status=status,
        stop_reason=_stop_reason(s, request, status),
    )
