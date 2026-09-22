"""Consistency checks on an answer file beyond what the schema can express.

Errors are things the README says score zero or break policy. Warnings are things that
are probably wrong but can be legitimate. Dataset checks (IDs exist, exposure adds up)
run only when a `DatasetIndex` is supplied.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol

from fraudtrail.answer.schema import ActionItem, Answer
from fraudtrail.domain import CaseStatus, Pattern, Trigger, Verdict
from fraudtrail.policy import constants as c
from fraudtrail.policy.actions import EXECUTION_ORDER, Action, route_for

# Answer Format, Part 2: the SAR narrative is "Six to twelve sentences".
SAR_MIN_SENTENCES = 6
SAR_MAX_SENTENCES = 12

# Answer Format, Part 2: activity_dates is [first, last].
SAR_DATE_COUNT = 2

# Tolerance when comparing dollar sums, to absorb float rounding of cent amounts.
USD_TOLERANCE = 0.01

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


class Level(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    level: Level
    field: str
    message: str


class DatasetIndex(Protocol):
    """Lookups the validator needs from the loaded dataset and graph."""

    def txn_amounts(self, txn_ids: Iterable[str]) -> dict[str, float]:
        """Amount in USD for each transaction ID that exists."""
        ...

    def unknown_ids(self, ids: Iterable[str]) -> set[str]:
        """IDs that match no customer, card, transaction, closed case, device profile or
        graph case."""
        ...


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(_SENTENCE_BREAK.split(stripped))


def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return len(value) == len("YYYY-MM-DD")


def _names(items: list[ActionItem]) -> list[Action]:
    return [item.action for item in items]


def _check_actions(answer: Answer, issues: list[Issue]) -> None:
    exposure = answer.case.exposure_usd
    nba = answer.next_best_actions
    for phase, items in (("initial", nba.initial), ("final", nba.final)):
        names = _names(items)
        if len(names) != len(set(names)):
            issues.append(Issue(Level.ERROR, f"next_best_actions.{phase}", "duplicate action"))
        for i, item in enumerate(items):
            expected = route_for(item.action, exposure)
            if item.route is not expected:
                issues.append(
                    Issue(
                        Level.ERROR,
                        f"next_best_actions.{phase}[{i}].route",
                        f"{item.action.value} at exposure ${exposure:,.2f} needs route "
                        f"{expected.value}, not {item.route.value}",
                    )
                )
            if not re.search(r"\bR\d+\b|\b\d[ab]?:", item.reason):
                issues.append(
                    Issue(
                        Level.WARNING,
                        f"next_best_actions.{phase}[{i}].reason",
                        "reason cites no policy rule (section 7 requires one)",
                    )
                )
        order = [EXECUTION_ORDER.index(a) for a in names]
        if order != sorted(order):
            issues.append(
                Issue(
                    Level.WARNING,
                    f"next_best_actions.{phase}",
                    "actions are not in execution order",
                )
            )

    final = set(_names(nba.final))
    if Action.FILE_REPORT in final and Action.CREATE_CASE not in final:
        issues.append(
            Issue(
                Level.ERROR, "next_best_actions.final", "3a: a report always has a case behind it"
            )
        )
    if Action.CLOSE_NO_FRAUD in final and final & {
        Action.BLOCK_CARD,
        Action.BLOCK_ALL_CARDS,
        Action.FILE_REPORT,
    }:
        issues.append(
            Issue(
                Level.ERROR,
                "next_best_actions.final",
                "CLOSE_NO_FRAUD alongside a block or a report",
            )
        )

    if not answer.evidence_requests:
        if _names(nba.initial) != _names(nba.final):
            issues.append(
                Issue(
                    Level.ERROR,
                    "next_best_actions.final",
                    "no evidence was requested, so final must equal initial",
                )
            )
        if nba.what_changed != "nothing":
            issues.append(
                Issue(
                    Level.ERROR,
                    "next_best_actions.what_changed",
                    'no evidence was requested, so what_changed must be "nothing"',
                )
            )


def _check_sar(answer: Answer, issues: list[Issue]) -> None:
    sar = answer.sar
    final = set(_names(answer.next_best_actions.final))
    if sar.file != (Action.FILE_REPORT in final):
        issues.append(
            Issue(
                Level.ERROR, "sar.file", "must agree with whether FILE_REPORT is in final actions"
            )
        )
    if not sar.file:
        empty = (
            sar.narrative == ""
            and sar.subjects == []
            and sar.total_amount_usd == 0
            and sar.activity_dates == []
        )
        if not empty:
            issues.append(
                Issue(
                    Level.ERROR,
                    "sar",
                    "when file is false: narrative, subjects, total and dates must be empty",
                )
            )
        return

    sentences = count_sentences(sar.narrative)
    if not SAR_MIN_SENTENCES <= sentences <= SAR_MAX_SENTENCES:
        issues.append(
            Issue(
                Level.ERROR,
                "sar.narrative",
                f"{sentences} sentences; the format requires {SAR_MIN_SENTENCES} to "
                f"{SAR_MAX_SENTENCES}",
            )
        )
    if not sar.subjects:
        issues.append(Issue(Level.ERROR, "sar.subjects", "a filed report must name its subjects"))
    if sar.total_amount_usd <= 0:
        issues.append(Issue(Level.ERROR, "sar.total_amount_usd", "a filed report needs a total"))
    if len(sar.activity_dates) != SAR_DATE_COUNT or not all(
        _is_iso_date(d) for d in sar.activity_dates
    ):
        issues.append(
            Issue(Level.ERROR, "sar.activity_dates", "must be [first, last] as YYYY-MM-DD")
        )
    elif sar.activity_dates[0] > sar.activity_dates[1]:
        issues.append(Issue(Level.ERROR, "sar.activity_dates", "first date is after last date"))
    if abs(sar.total_amount_usd - answer.case.exposure_usd) > USD_TOLERANCE:
        issues.append(
            Issue(
                Level.WARNING,
                "sar.total_amount_usd",
                f"${sar.total_amount_usd:,.2f} differs from case exposure "
                f"${answer.case.exposure_usd:,.2f}",
            )
        )


def _check_case(answer: Answer, issues: list[Issue]) -> None:
    case = answer.case
    if case.verdict is Verdict.LEGITIMATE:
        if case.affected_txn_ids or case.exposure_usd != 0 or case.first_suspicious_txn_id:
            issues.append(
                Issue(
                    Level.ERROR,
                    "case",
                    "legitimate verdict: affected_txn_ids and first_suspicious_txn_id must be "
                    "empty and exposure 0",
                )
            )
        if answer.sar.file:
            issues.append(Issue(Level.ERROR, "sar.file", "legitimate verdict cannot file a report"))

    if case.pattern is Pattern.UNDOCUMENTED and not case.pattern_description.strip():
        issues.append(
            Issue(Level.ERROR, "case.pattern_description", "required when pattern is undocumented")
        )
    if case.pattern is not Pattern.UNDOCUMENTED and case.pattern_description != "":
        issues.append(
            Issue(
                Level.ERROR, "case.pattern_description", 'must be "" unless pattern is undocumented'
            )
        )

    if case.first_suspicious_txn_id and case.first_suspicious_txn_id not in case.affected_txn_ids:
        issues.append(
            Issue(Level.ERROR, "case.first_suspicious_txn_id", "not among affected_txn_ids")
        )
    if case.affected_txn_ids and not case.first_suspicious_txn_id:
        issues.append(
            Issue(Level.WARNING, "case.first_suspicious_txn_id", "empty although txns are affected")
        )
    if len(case.affected_txn_ids) != len(set(case.affected_txn_ids)):
        issues.append(Issue(Level.ERROR, "case.affected_txn_ids", "duplicate transaction IDs"))

    if case.written_to_graph != bool(case.graph_case_id):
        issues.append(
            Issue(
                Level.ERROR,
                "case.graph_case_id",
                "must be set exactly when written_to_graph is true",
            )
        )
    if not case.evidence:
        issues.append(Issue(Level.WARNING, "case.evidence", "no evidence listed"))

    final = set(_names(answer.next_best_actions.final))
    consistent = {
        CaseStatus.CLOSED_LEGITIMATE: case.verdict is Verdict.LEGITIMATE,
        CaseStatus.CLOSED_FRAUD: case.verdict is Verdict.FRAUD,
        CaseStatus.ESCALATED: Action.ESCALATE_TO_ANALYST in final,
    }
    if not consistent.get(case.status, True):
        issues.append(
            Issue(
                Level.ERROR,
                "case.status",
                f"status {case.status.value} contradicts verdict {case.verdict.value} or the "
                f"final actions",
            )
        )
    if Action.CLOSE_NO_FRAUD in final and case.verdict is not Verdict.LEGITIMATE:
        issues.append(
            Issue(Level.ERROR, "case.verdict", "CLOSE_NO_FRAUD requires a legitimate verdict")
        )

    steps = [r.asked_after_step for r in answer.evidence_requests]
    if steps != sorted(steps):
        issues.append(
            Issue(Level.WARNING, "evidence_requests", "asked_after_step is not in increasing order")
        )


def _check_case_opening(answer: Answer, trigger: Trigger, issues: list[Issue]) -> None:
    """3a: a case opens on a dispute, on any evidence request, or at probability 0.30."""
    nba = answer.next_best_actions
    anywhere = set(_names(nba.initial)) | set(_names(nba.final))
    needs_case = (
        trigger is Trigger.CUSTOMER_REPORT
        or bool(answer.evidence_requests)
        or answer.case.fraud_probability >= c.CASE_OPEN_MIN_PROBABILITY
    )
    if needs_case and Action.CREATE_CASE not in anywhere:
        issues.append(
            Issue(
                Level.ERROR,
                "next_best_actions",
                "3a requires CREATE_CASE (dispute, evidence request, or probability >= 0.30)",
            )
        )


def _check_dataset(answer: Answer, data: DatasetIndex, issues: list[Issue]) -> None:
    case = answer.case
    ids: set[str] = set(case.affected_txn_ids)
    ids |= set(case.connected_card_ids)
    ids |= set(case.connected_device_profiles)
    ids |= set(case.similar_prior_cases)
    ids |= set(answer.sar.subjects)
    for ev in case.evidence:
        ids |= set(ev.entity_ids)
    if case.first_suspicious_txn_id:
        ids.add(case.first_suspicious_txn_id)
    unknown = data.unknown_ids(ids)
    if unknown:
        issues.append(Issue(Level.ERROR, "ids", f"IDs not in the dataset: {sorted(unknown)}"))

    amounts = data.txn_amounts(case.affected_txn_ids)
    expected = sum(abs(amounts[t]) for t in case.affected_txn_ids if t in amounts)
    if abs(expected - case.exposure_usd) > USD_TOLERANCE:
        issues.append(
            Issue(
                Level.ERROR,
                "case.exposure_usd",
                f"${case.exposure_usd:,.2f} but affected transactions sum to ${expected:,.2f}",
            )
        )


def check_answer(
    answer: Answer,
    *,
    trigger: Trigger | None = None,
    data: DatasetIndex | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    _check_case(answer, issues)
    _check_actions(answer, issues)
    _check_sar(answer, issues)
    if trigger is not None:
        _check_case_opening(answer, trigger, issues)
    if data is not None:
        _check_dataset(answer, data, issues)
    return issues
