"""Scoring a replayed case against what the bank's analysts decided.

Six questions, because an agent can be right about one and wrong about the next: did it
reach the same verdict, name the same pattern, hold the same transactions responsible,
arrive at the same exposure, take the same actions, and agree about filing a report.

Nothing here rewards a confident guess. An uncertain verdict counts as its own outcome
rather than as a wrong answer, so an agent that says "I cannot tell" is not scored as if
it had decided, and the summary reports how often it happened.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from fraudtrail.answer.schema import Answer
from fraudtrail.domain import Verdict
from fraudtrail.evaluate.replay import ClosedCaseTruth

# Exposure within this fraction of the recorded amount counts as agreement: the bank's
# figure is the sum of the transactions it held responsible, so a match means the same
# transactions, not a lucky total.
EXPOSURE_TOLERANCE = 0.01

# Actions the bank records but this agent has no way to reach in a replay: the history
# names what was eventually done, including steps taken after the case closed.
UNSCORED_ACTIONS = frozenset({"MONITOR_CARD", "MONITOR_CONNECTED_CARDS", "GENERATE_REPORT"})


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    truth_fraud: bool
    predicted: Verdict
    truth_pattern: str
    predicted_pattern: str
    txn_recall: float
    txn_precision: float
    exposure_truth: float
    exposure_predicted: float
    action_f1: float
    report_truth: bool
    report_predicted: bool

    @property
    def verdict_correct(self) -> bool:
        if self.predicted is Verdict.UNCERTAIN:
            return False
        return (self.predicted is Verdict.FRAUD) == self.truth_fraud

    @property
    def pattern_correct(self) -> bool:
        return self.truth_fraud and self.predicted_pattern == self.truth_pattern

    @property
    def exposure_correct(self) -> bool:
        if self.exposure_truth == 0:
            return self.exposure_predicted == 0
        gap = abs(self.exposure_predicted - self.exposure_truth) / self.exposure_truth
        return gap <= EXPOSURE_TOLERANCE

    @property
    def report_correct(self) -> bool:
        return self.report_truth == self.report_predicted


def _f1(predicted: set[str], truth: set[str]) -> tuple[float, float, float]:
    if not predicted and not truth:
        return 1.0, 1.0, 1.0
    if not predicted or not truth:
        return 0.0, 0.0, 0.0
    hit = len(predicted & truth)
    precision = hit / len(predicted)
    recall = hit / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    return precision, recall, f1


def score_case(truth: ClosedCaseTruth, answer: Answer) -> CaseScore:
    predicted_txns = set(answer.case.affected_txn_ids)
    truth_txns = set(truth.txn_ids) if truth.is_fraud else set()
    precision, recall, _ = _f1(predicted_txns, truth_txns)

    predicted_actions = {a.action.value.upper() for a in answer.next_best_actions.final}
    truth_actions = {a.upper() for a in truth.actions}
    scored_predicted = predicted_actions - UNSCORED_ACTIONS
    scored_truth = truth_actions - UNSCORED_ACTIONS
    _, _, action_f1 = _f1(scored_predicted, scored_truth)

    return CaseScore(
        case_id=truth.case_id,
        truth_fraud=truth.is_fraud,
        predicted=answer.case.verdict,
        truth_pattern=truth.pattern,
        predicted_pattern=answer.case.pattern.value,
        txn_recall=recall,
        txn_precision=precision,
        exposure_truth=truth.exposure_usd,
        exposure_predicted=answer.case.exposure_usd,
        action_f1=action_f1,
        report_truth=truth.report_filed,
        report_predicted=answer.sar.file,
    )


@dataclass
class Summary:
    """Totals over every replayed case."""

    scores: list[CaseScore] = field(default_factory=list)

    def add(self, score: CaseScore) -> None:
        self.scores.append(score)

    @property
    def n(self) -> int:
        return len(self.scores)

    def _share(self, predicate: Sequence[bool]) -> float:
        return sum(predicate) / len(predicate) if predicate else 0.0

    @property
    def verdict_accuracy(self) -> float:
        return self._share([s.verdict_correct for s in self.scores])

    @property
    def uncertain_share(self) -> float:
        return self._share([s.predicted is Verdict.UNCERTAIN for s in self.scores])

    @property
    def fraud_recall(self) -> float:
        """Of the cases the bank confirmed as fraud, how many the agent also called fraud."""
        fraud = [s for s in self.scores if s.truth_fraud]
        return self._share([s.predicted is Verdict.FRAUD for s in fraud])

    @property
    def cleared_recall(self) -> float:
        """Of the alerts the bank cleared, how many the agent also cleared."""
        cleared = [s for s in self.scores if not s.truth_fraud]
        return self._share([s.predicted is Verdict.LEGITIMATE for s in cleared])

    @property
    def pattern_accuracy(self) -> float:
        """Pattern agreement where both sides say fraud; naming one is only then meaningful."""
        both = [s for s in self.scores if s.truth_fraud and s.predicted is Verdict.FRAUD]
        return self._share([s.pattern_correct for s in both])

    @property
    def txn_recall(self) -> float:
        fraud = [s for s in self.scores if s.truth_fraud]
        return sum(s.txn_recall for s in fraud) / len(fraud) if fraud else 0.0

    @property
    def txn_precision(self) -> float:
        fraud = [s for s in self.scores if s.truth_fraud]
        return sum(s.txn_precision for s in fraud) / len(fraud) if fraud else 0.0

    @property
    def exposure_accuracy(self) -> float:
        fraud = [s for s in self.scores if s.truth_fraud]
        return self._share([s.exposure_correct for s in fraud])

    @property
    def action_f1(self) -> float:
        return sum(s.action_f1 for s in self.scores) / self.n if self.n else 0.0

    @property
    def report_agreement(self) -> float:
        return self._share([s.report_correct for s in self.scores])

    @property
    def report_recall(self) -> float:
        filed = [s for s in self.scores if s.report_truth]
        return self._share([s.report_predicted for s in filed])

    @property
    def false_report_rate(self) -> float:
        """Reports the agent would file that the bank did not: the expensive mistake."""
        not_filed = [s for s in self.scores if not s.report_truth]
        return self._share([s.report_predicted for s in not_filed])

    def pattern_confusion(self) -> dict[tuple[str, str], int]:
        both = [s for s in self.scores if s.truth_fraud and s.predicted is Verdict.FRAUD]
        confusion: dict[tuple[str, str], int] = {}
        for s in both:
            key = (s.truth_pattern, s.predicted_pattern)
            confusion[key] = confusion.get(key, 0) + 1
        return confusion
