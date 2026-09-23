"""The 20 exam cases, read from case_pack.csv."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fraudtrail.domain import Trigger

TIMESTAMP = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class ExamCase:
    case_id: str
    opened_at: datetime
    trigger: Trigger
    trigger_text: str
    flagged_txn_id: str
    card_id: str
    customer_id: str
    risk_score: float | None

    @property
    def flagged_is_pending(self) -> bool:
        """A real-time model alert fires on an authorization that can still be stopped.

        A customer report or an analyst request arrives after the purchase settled.
        """
        return self.trigger is Trigger.RISK_SCORE


def load_case_pack(path: Path) -> tuple[ExamCase, ...]:
    """Every exam case, oldest first, so earlier investigations become memory for later ones."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cases = [
        ExamCase(
            case_id=row["case_id"].strip(),
            opened_at=datetime.strptime(row["opened_at"].strip(), TIMESTAMP),
            trigger=Trigger(row["trigger_type"].strip()),
            trigger_text=row["trigger_text"].strip(),
            flagged_txn_id=row["flagged_txn_id"].strip(),
            card_id=row["card_id"].strip(),
            customer_id=row["customer_id"].strip(),
            risk_score=float(row["risk_score"]) if row.get("risk_score", "").strip() else None,
        )
        for row in rows
    ]
    return tuple(sorted(cases, key=lambda case: case.opened_at))
