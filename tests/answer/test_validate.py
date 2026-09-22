import copy
from collections.abc import Iterable
from typing import Any

import pytest

from fraudtrail.answer.schema import Answer
from fraudtrail.answer.validate import Issue, Level, check_answer, count_sentences
from fraudtrail.domain import Trigger

NARRATIVE = (
    "On 2016-11-02 card C00001-K1 of customer C00001 made two online purchases of $100.00 "
    "and $50.00. "
    "Both came from a device profile marked New for this account. "
    "The same device profile was used on card C00001-K2 on 2016-11-03. "
    "The cardholder denied both purchases when contacted. "
    "The pattern is consistent with card-not-present use of a compromised card number. "
    "Total suspicious amount: $150.00. "
    "The card was blocked and the connected card placed under monitoring."
)


def valid_answer() -> dict[str, Any]:
    return {
        "case_id": "HHG-900",
        "case": {
            "status": "closed_fraud",
            "verdict": "fraud",
            "fraud_probability": 0.9,
            "pattern": "card_not_present_new_device",
            "pattern_description": "",
            "affected_txn_ids": ["3400001", "3400002"],
            "first_suspicious_txn_id": "3400001",
            "connected_card_ids": ["C00001-K2"],
            "connected_device_profiles": ["Device X | OS Y | Browser Z | 100x100"],
            "exposure_usd": 150.0,
            "evidence": [
                {
                    "claim": "Two online purchases from a new device within an hour",
                    "source": "graph",
                    "ref": "query:card_window(card_id=C00001-K1, hours=2)",
                    "entity_ids": ["3400001", "3400002"],
                }
            ],
            "similar_prior_cases": ["CC-0001"],
            "summary": "Two card-not-present purchases from a new device; customer denied.",
            "written_to_graph": True,
            "graph_case_id": "CASE-900",
        },
        "evidence_requests": [
            {
                "type": "customer_validation",
                "asked_after_step": 3,
                "assumed_response": "Customer states they did not make these purchases",
            }
        ],
        "next_best_actions": {
            "initial": [
                {"action": "VERIFY_WITH_CUSTOMER", "route": "auto", "reason": "R1: verify first"},
                {"action": "CREATE_CASE", "route": "auto", "reason": "3a: evidence requested"},
            ],
            "final": [
                {"action": "BLOCK_CARD", "route": "L1", "reason": "R2: customer denied"},
                {"action": "CREATE_CASE", "route": "auto", "reason": "R2"},
                {"action": "FILE_REPORT", "route": "L2", "reason": "3a: shared device"},
                {"action": "MONITOR_CONNECTED_CARDS", "route": "auto", "reason": "R6"},
            ],
            "what_changed": "Denial raised probability and the shared device requires a report.",
        },
        "sar": {
            "file": True,
            "reason": "3a: shared device links to a second card",
            "narrative": NARRATIVE,
            "subjects": ["C00001", "C00001-K1", "C00001-K2"],
            "total_amount_usd": 150.0,
            "activity_dates": ["2016-11-02", "2016-11-03"],
        },
        "stop_reason": "Denial settled the verdict.",
        "tool_calls": 5,
        "tokens": 1000,
        "latency_s": 3.2,
    }


def errors(raw: dict[str, Any], **kwargs: Any) -> list[Issue]:
    return [i for i in check_answer(Answer.model_validate(raw), **kwargs) if i.level is Level.ERROR]


def error_fields(raw: dict[str, Any], **kwargs: Any) -> set[str]:
    return {i.field for i in errors(raw, **kwargs)}


class FakeIndex:
    def __init__(self, amounts: dict[str, float], known: set[str]) -> None:
        self._amounts = amounts
        self._known = known

    def txn_amounts(self, txn_ids: Iterable[str]) -> dict[str, float]:
        return {t: self._amounts[t] for t in txn_ids if t in self._amounts}

    def unknown_ids(self, ids: Iterable[str]) -> set[str]:
        return set(ids) - self._known


def test_valid_answer_has_no_issues() -> None:
    assert check_answer(Answer.model_validate(valid_answer()), trigger=Trigger.RISK_SCORE) == []


def test_schema_rejects_unknown_action_and_extra_fields() -> None:
    raw = valid_answer()
    raw["next_best_actions"]["final"][0]["action"] = "FREEZE_ACCOUNT"
    with pytest.raises(ValueError):
        Answer.model_validate(raw)
    raw = valid_answer()
    raw["case"]["notes"] = "extra"
    with pytest.raises(ValueError):
        Answer.model_validate(raw)


def test_sar_flag_must_match_file_report() -> None:
    raw = valid_answer()
    raw["sar"]["file"] = False
    assert "sar.file" in error_fields(raw)


def test_wrong_route_is_an_error() -> None:
    raw = valid_answer()
    raw["next_best_actions"]["final"][0]["route"] = "L2"
    assert "next_best_actions.final[0].route" in error_fields(raw)


def test_legitimate_verdict_must_have_no_affected_transactions() -> None:
    raw = valid_answer()
    raw["case"]["verdict"] = "legitimate"
    assert "case" in error_fields(raw)


def test_narrative_needs_six_to_twelve_sentences() -> None:
    raw = valid_answer()
    raw["sar"]["narrative"] = "Too short. Only two sentences."
    assert "sar.narrative" in error_fields(raw)


def test_without_evidence_requests_final_must_equal_initial() -> None:
    raw = valid_answer()
    raw["evidence_requests"] = []
    fields = error_fields(raw)
    assert {"next_best_actions.final", "next_best_actions.what_changed"} <= fields


def test_undocumented_pattern_needs_a_description() -> None:
    raw = valid_answer()
    raw["case"]["pattern"] = "undocumented"
    assert "case.pattern_description" in error_fields(raw)


def test_report_without_case_is_an_error() -> None:
    raw = valid_answer()
    final = raw["next_best_actions"]["final"]
    raw["next_best_actions"]["final"] = [a for a in final if a["action"] != "CREATE_CASE"]
    assert "next_best_actions.final" in error_fields(raw)


def test_dispute_requires_a_case() -> None:
    raw = valid_answer()
    raw["evidence_requests"] = []
    raw["case"]["fraud_probability"] = 0.1
    only = [{"action": "VERIFY_WITH_CUSTOMER", "route": "auto", "reason": "R7"}]
    raw["next_best_actions"]["initial"] = copy.deepcopy(only)
    raw["next_best_actions"]["final"] = copy.deepcopy(only)
    raw["next_best_actions"]["what_changed"] = "nothing"
    messages = [i.message for i in errors(raw, trigger=Trigger.CUSTOMER_REPORT)]
    assert any("CREATE_CASE" in m for m in messages)


def test_dataset_checks_ids_and_exposure() -> None:
    raw = valid_answer()
    known = {
        "3400001",
        "3400002",
        "C00001",
        "C00001-K1",
        "C00001-K2",
        "CC-0001",
        "Device X | OS Y | Browser Z | 100x100",
    }
    index = FakeIndex({"3400001": 100.0, "3400002": -50.0}, known)
    assert errors(raw, data=index) == []

    raw["case"]["exposure_usd"] = 120.0
    raw["sar"]["total_amount_usd"] = 120.0
    assert "case.exposure_usd" in error_fields(raw, data=index)

    raw = valid_answer()
    raw["case"]["similar_prior_cases"] = ["CC-9999"]
    assert "ids" in error_fields(raw, data=index)


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("", 0),
        ("Amounts were $1.10 and $2.40 on card C00001-K1.", 1),
        ("One. Two! Three? 2016-11-02 was the fourth.", 4),
        (NARRATIVE, 7),
    ],
)
def test_count_sentences(text: str, count: int) -> None:
    assert count_sentences(text) == count
