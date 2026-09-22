from dataclasses import replace

import pytest

from fraudtrail.domain import CaseStatus, Trigger, Verification
from fraudtrail.policy.actions import Action, Route
from fraudtrail.policy.engine import Decision, Situation, decide

A = Action


def alert(**overrides: object) -> Situation:
    base = Situation(
        trigger=Trigger.RISK_SCORE,
        fraud_probability=0.45,
        independent_evidence=1,
        exposure_usd=100.0,
        flagged_is_pending=True,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def dispute(**overrides: object) -> Situation:
    return alert(trigger=Trigger.CUSTOMER_REPORT, flagged_is_pending=False, **overrides)


def routes(decision: Decision) -> dict[Action, Route]:
    return {r.action: r.route for r in decision.actions}


class TestR1AndSection3bExample:
    """README 3b: 0.45 on one signal -> verify; denial -> block and case."""

    def test_initial_verifies_and_opens_a_case_without_blocking(self) -> None:
        d = decide(alert())
        assert d.action_names == [A.VERIFY_WITH_CUSTOMER, A.CREATE_CASE]
        assert d.evidence_request is A.VERIFY_WITH_CUSTOMER
        assert d.status is CaseStatus.OPEN
        assert "R1" in d.actions[0].reason

    def test_final_after_denial_blocks(self) -> None:
        after = alert(
            fraud_probability=0.90,
            independent_evidence=2,
            verification=Verification.DENIED,
            case_opened=True,
        )
        d = decide(after)
        assert d.action_names == [A.BLOCK_CARD, A.CREATE_CASE]
        assert routes(d)[A.BLOCK_CARD] is Route.L1
        assert d.evidence_request is None
        assert d.status is CaseStatus.CLOSED_FRAUD
        assert not d.report.file

    def test_single_signal_never_blocks_even_on_a_dispute(self) -> None:
        d = decide(dispute(fraud_probability=0.5, independent_evidence=1))
        assert A.BLOCK_CARD not in d.action_names
        assert d.evidence_request is A.VERIFY_WITH_CUSTOMER


class TestR2CustomerDenies:
    def test_report_above_1000_and_route_by_exposure(self) -> None:
        d = decide(dispute(fraud_probability=0.9, independent_evidence=3, exposure_usd=1_500.0))
        assert d.action_names == [A.BLOCK_CARD, A.CREATE_CASE, A.FILE_REPORT]
        assert routes(d)[A.BLOCK_CARD] is Route.L1
        assert routes(d)[A.FILE_REPORT] is Route.L2
        assert d.report.file

    def test_block_needs_l2_above_2500(self) -> None:
        d = decide(dispute(fraud_probability=0.9, independent_evidence=3, exposure_usd=3_000.0))
        assert routes(d)[A.BLOCK_CARD] is Route.L2

    def test_link_to_other_fraud_files_and_monitors_in_readme_order(self) -> None:
        d = decide(
            alert(
                fraud_probability=0.86,
                independent_evidence=3,
                exposure_usd=268.43,
                verification=Verification.DENIED,
                linked_to_other_fraud=True,
                connected_cards=1,
                case_opened=True,
            )
        )
        assert d.action_names == [
            A.BLOCK_CARD,
            A.CREATE_CASE,
            A.FILE_REPORT,
            A.MONITOR_CONNECTED_CARDS,
        ]


def test_r3_confirmation_closes() -> None:
    d = decide(
        alert(
            fraud_probability=0.05,
            independent_evidence=2,
            verification=Verification.CONFIRMED,
            case_opened=True,
        )
    )
    assert d.action_names == [A.CREATE_CASE, A.ALLOW_TRANSACTION, A.CLOSE_NO_FRAUD]
    assert d.status is CaseStatus.CLOSED_LEGITIMATE
    assert not d.report.file


def test_r4_no_reply_monitors_declines_and_escalates_above_500() -> None:
    d = decide(
        alert(
            fraud_probability=0.5,
            exposure_usd=600.0,
            verification=Verification.NO_REPLY,
            case_opened=True,
        )
    )
    assert d.action_names == [
        A.DECLINE_TRANSACTION,
        A.CREATE_CASE,
        A.MONITOR_CARD,
        A.ESCALATE_TO_ANALYST,
    ]
    assert d.status is CaseStatus.ESCALATED


class TestR5CardTesting:
    def test_cleared_purchase_over_100_blocks(self) -> None:
        d = decide(
            alert(
                fraud_probability=0.8,
                independent_evidence=2,
                exposure_usd=268.43,
                card_testing=True,
                card_testing_cleared_purchase_usd=259.98,
            )
        )
        assert d.action_names == [
            A.DECLINE_TRANSACTION,
            A.BLOCK_CARD,
            A.STEP_UP_AUTH,
            A.CREATE_CASE,
        ]
        assert d.evidence_request is A.STEP_UP_AUTH

    def test_without_cleared_purchase_declines_and_steps_up(self) -> None:
        d = decide(alert(fraud_probability=0.8, independent_evidence=2, card_testing=True))
        assert A.BLOCK_CARD not in d.action_names
        assert d.action_names[:2] == [A.DECLINE_TRANSACTION, A.STEP_UP_AUTH]


def test_r6_shared_origin_files_and_monitors_connected_cards() -> None:
    d = decide(
        alert(
            trigger=Trigger.ANALYST_REQUEST,
            fraud_probability=0.8,
            independent_evidence=3,
            exposure_usd=400.0,
            flagged_is_pending=False,
            shared_origin=True,
            connected_cards=3,
        )
    )
    assert {A.CREATE_CASE, A.FILE_REPORT, A.MONITOR_CONNECTED_CARDS} <= set(d.action_names)
    assert d.report.file
    assert "R6" in d.report.reason


class TestR7RecurringDispute:
    def test_initial_verifies_and_warns_without_blocking(self) -> None:
        d = decide(
            dispute(
                fraud_probability=0.2,
                independent_evidence=2,
                exposure_usd=49.0,
                recurring_charge_match=True,
            )
        )
        assert d.action_names == [A.VERIFY_WITH_CUSTOMER, A.CREATE_CASE, A.WARN_CUSTOMER]
        assert d.evidence_request is A.VERIFY_WITH_CUSTOMER

    def test_final_after_confirmation_closes(self) -> None:
        d = decide(
            dispute(
                fraud_probability=0.05,
                independent_evidence=3,
                recurring_charge_match=True,
                verification=Verification.CONFIRMED,
                case_opened=True,
            )
        )
        assert d.action_names == [A.CREATE_CASE, A.WARN_CUSTOMER, A.CLOSE_NO_FRAUD]
        assert d.status is CaseStatus.CLOSED_LEGITIMATE


def test_r8_uncertain_and_exposed_escalates() -> None:
    d = decide(alert(fraud_probability=0.5, independent_evidence=2, exposure_usd=800.0))
    assert A.ESCALATE_TO_ANALYST in d.action_names
    assert d.status is CaseStatus.ESCALATED


def test_r9_undocumented_coordinated_files_and_escalates() -> None:
    d = decide(
        alert(
            trigger=Trigger.ANALYST_REQUEST,
            fraud_probability=0.8,
            independent_evidence=3,
            coordinated_undocumented=True,
        )
    )
    assert {A.CREATE_CASE, A.FILE_REPORT, A.ESCALATE_TO_ANALYST} <= set(d.action_names)


def test_undocumented_single_card_pattern_reports_without_escalating() -> None:
    """Closed cases CC-3748 and similar: under-$500 bursts were filed but not escalated."""
    d = decide(
        dispute(
            fraud_probability=0.9,
            independent_evidence=3,
            exposure_usd=450.0,
            undocumented_pattern=True,
        )
    )
    assert d.action_names == [A.BLOCK_CARD, A.CREATE_CASE, A.FILE_REPORT]
    assert "undocumented" in d.report.reason


@pytest.mark.parametrize(
    ("confirmed_cards", "expected"), [(1, A.BLOCK_CARD), (2, A.BLOCK_ALL_CARDS)]
)
def test_r10_block_all_cards_needs_two_confirmed_cards(
    confirmed_cards: int, expected: Action
) -> None:
    d = decide(
        dispute(
            fraud_probability=0.9,
            independent_evidence=3,
            cards_with_confirmed_fraud=confirmed_cards,
        )
    )
    blocks = {A.BLOCK_CARD, A.BLOCK_ALL_CARDS} & set(d.action_names)
    assert blocks == {expected}


def test_decisive_legitimate_alert_closes_without_a_case() -> None:
    d = decide(alert(fraud_probability=0.08, independent_evidence=3, exposure_usd=0.0))
    assert d.action_names == [A.ALLOW_TRANSACTION, A.CLOSE_NO_FRAUD]
    assert d.status is CaseStatus.CLOSED_LEGITIMATE
    assert d.evidence_request is None


def test_decisive_fraud_alert_stops_and_reports_above_1000() -> None:
    d = decide(alert(fraud_probability=0.9, independent_evidence=3, exposure_usd=1_200.0))
    assert d.action_names == [A.BLOCK_CARD, A.CREATE_CASE, A.FILE_REPORT]
    assert d.status is CaseStatus.CLOSED_FRAUD
    assert d.evidence_request is None


def test_exposure_report_waits_for_requested_evidence() -> None:
    d = decide(alert(fraud_probability=0.75, independent_evidence=2, exposure_usd=1_500.0))
    assert d.evidence_request is A.VERIFY_WITH_CUSTOMER
    assert not d.report.file
    assert "3b" in d.report.reason


def test_dispute_contradicted_by_evidence_is_verified_not_blocked() -> None:
    d = decide(dispute(fraud_probability=0.1, independent_evidence=3))
    assert A.BLOCK_CARD not in d.action_names
    assert d.evidence_request is A.VERIFY_WITH_CUSTOMER


def test_situation_rejects_out_of_range_inputs() -> None:
    with pytest.raises(ValueError):
        alert(fraud_probability=1.2)
    with pytest.raises(ValueError):
        alert(exposure_usd=-1.0)
