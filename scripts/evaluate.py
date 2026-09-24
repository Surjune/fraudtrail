"""Replay the bank's closed cases through the agent and score the result.

The exam's answer key is not public, so this is the only measurement available: run the
same investigation over cases the analysts already decided, with the case under test
hidden, and compare. Sampling is stratified and seeded, so two runs of the same seed
measure the same thing and a change in the score is a change in the agent.

Usage::

    uv run python scripts/evaluate.py [--n 120] [--seed 7] [--offline] [--out docs/evaluation.md]
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from fraudtrail.answer.build import build_answer
from fraudtrail.answer.narration import TemplateNarrator
from fraudtrail.config import load_settings
from fraudtrail.evaluate.metrics import CaseScore, Summary, score_case
from fraudtrail.evaluate.replay import ClosedCaseTruth, HoldOutProvider, load_closed_cases
from fraudtrail.evidence.duckdb_provider import DuckDbProvider
from fraudtrail.evidence.provider import EvidenceError, EvidenceProvider
from fraudtrail.evidence.tigergraph_provider import TigerGraphProvider
from fraudtrail.investigate.runner import investigate

log = logging.getLogger("evaluate")

REPO = Path(__file__).resolve().parent.parent

# Half fraud, half cleared, whatever the history's own balance is: the exam pack is
# deliberately balanced, and a score dominated by one class would say nothing about the
# other.
FRAUD_SHARE = 0.5


def sample(cases: tuple[ClosedCaseTruth, ...], n: int, seed: int) -> list[ClosedCaseTruth]:
    fraud = [c for c in cases if c.is_fraud]
    cleared = [c for c in cases if not c.is_fraud]
    rng = random.Random(seed)
    want_fraud = min(int(n * FRAUD_SHARE), len(fraud))
    want_cleared = min(n - want_fraud, len(cleared))
    chosen = rng.sample(fraud, want_fraud) + rng.sample(cleared, want_cleared)
    chosen.sort(key=lambda c: c.opened_at)
    return chosen


def write_report(summary: Summary, path: Path, seed: int, source: str) -> None:
    lines = [
        "# Replay evaluation",
        "",
        f"{summary.n} closed cases replayed from {source}, seed {seed}. The case under "
        "test is hidden from every query, so the agent cannot retrieve its own answer.",
        "",
        "| Measure | Result |",
        "| --- | --- |",
        f"| Verdict accuracy | {summary.verdict_accuracy:.1%} |",
        f"| Confirmed fraud called fraud | {summary.fraud_recall:.1%} |",
        f"| Cleared alerts cleared | {summary.cleared_recall:.1%} |",
        f"| Verdict left uncertain | {summary.uncertain_share:.1%} |",
        f"| Pattern named correctly | {summary.pattern_accuracy:.1%} |",
        f"| Suspicious transactions found | {summary.txn_recall:.1%} |",
        f"| Suspicious transactions precision | {summary.txn_precision:.1%} |",
        f"| Exposure matched | {summary.exposure_accuracy:.1%} |",
        f"| Action agreement (F1) | {summary.action_f1:.1%} |",
        f"| Report decision agreement | {summary.report_agreement:.1%} |",
        f"| Reports the bank filed, also filed | {summary.report_recall:.1%} |",
        f"| Reports filed the bank did not | {summary.false_report_rate:.1%} |",
        "",
        "## Pattern confusion",
        "",
        "| The bank said | The agent said | Cases |",
        "| --- | --- | --- |",
    ]
    confusion = sorted(summary.pattern_confusion().items(), key=lambda kv: -kv[1])
    lines.extend(f"| {truth} | {predicted} | {count} |" for (truth, predicted), count in confusion)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_provider(offline: bool) -> EvidenceProvider:
    settings = load_settings()
    if offline:
        return DuckDbProvider.open(settings.raw_dir, settings.processed_dir)
    return TigerGraphProvider.from_env(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=120, help="how many cases to replay")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--offline", action="store_true", help="use the local warehouse")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "evaluation.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fraudtrail.investigate.runner").setLevel(logging.WARNING)

    settings = load_settings()
    history = load_closed_cases(settings.raw_dir / "closed_cases_history.csv")
    chosen = sample(history, args.n, args.seed)
    log.info("replaying %d of %d closed cases", len(chosen), len(history))

    provider = build_provider(args.offline)
    narrator = TemplateNarrator()
    summary = Summary()
    skipped = 0

    for index, truth in enumerate(chosen, start=1):
        try:
            result = investigate(HoldOutProvider(provider, truth.case_id), truth.as_exam_case())
        except EvidenceError as exc:
            # A case whose transaction is not in the dataset cannot be replayed; it is
            # left out of the score rather than counted as a failure.
            log.warning("%s skipped: %s", truth.case_id, exc)
            skipped += 1
            continue
        score: CaseScore = score_case(truth, build_answer(result, narrator))
        summary.add(score)
        if index % 20 == 0:
            log.info(
                "%d/%d replayed, verdict accuracy so far %.1f%%",
                index,
                len(chosen),
                summary.verdict_accuracy * 100,
            )

    source = "the local warehouse" if args.offline else "TigerGraph"
    write_report(summary, args.out, args.seed, source)
    log.info("skipped %d, scored %d", skipped, summary.n)
    log.info(
        "verdict accuracy %.1f%% (fraud %.1f%%, cleared %.1f%%, uncertain %.1f%%)",
        summary.verdict_accuracy * 100,
        summary.fraud_recall * 100,
        summary.cleared_recall * 100,
        summary.uncertain_share * 100,
    )
    log.info(
        "pattern %.1f%% | txn recall %.1f%% | exposure %.1f%% | actions %.1f%% | report %.1f%%",
        summary.pattern_accuracy * 100,
        summary.txn_recall * 100,
        summary.exposure_accuracy * 100,
        summary.action_f1 * 100,
        summary.report_agreement * 100,
    )
    log.info("report written to %s", args.out)


if __name__ == "__main__":
    main()
