"""Run the agent over the exam cases and write one answer file per case.

Cases run oldest first, so each investigation can retrieve the ones before it. Every file
is validated before it is written; a file with errors is written anyway and reported, so
the failure is visible rather than silent.

Usage::

    uv run python scripts/run_agent.py [--only HHG-014] [--out cases] [--offline]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from fraudtrail.answer.build import build_answer
from fraudtrail.answer.narration import TemplateNarrator
from fraudtrail.answer.validate import Level, check_answer
from fraudtrail.casepack import load_case_pack
from fraudtrail.config import load_settings
from fraudtrail.evidence.duckdb_provider import DuckDbProvider
from fraudtrail.investigate.runner import investigate

log = logging.getLogger("run_agent")

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", default="", help="comma-separated case IDs")
    parser.add_argument("--out", type=Path, default=REPO / "cases")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the local warehouse instead of TigerGraph",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    cases = load_case_pack(settings.raw_dir / "case_pack.csv")
    wanted = {c.strip() for c in args.only.split(",") if c.strip()}
    if wanted:
        cases = tuple(case for case in cases if case.case_id in wanted)

    provider = DuckDbProvider.open(settings.raw_dir, settings.processed_dir)
    narrator = TemplateNarrator()
    args.out.mkdir(parents=True, exist_ok=True)

    failures = 0
    for case in cases:
        result = investigate(provider, case)
        answer = build_answer(result, narrator)
        issues = check_answer(answer, trigger=case.trigger)
        errors = [i for i in issues if i.level is Level.ERROR]
        for issue in issues:
            log.log(
                logging.ERROR if issue.level is Level.ERROR else logging.WARNING,
                "%s %s: %s",
                case.case_id,
                issue.field,
                issue.message,
            )
        failures += len(errors)
        path = args.out / f"{case.case_id}.json"
        path.write_text(
            json.dumps(answer.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )

    log.info("wrote %d answer files to %s", len(cases), args.out)
    if failures:
        raise SystemExit(f"{failures} validation error(s); fix the agent, not the files")


if __name__ == "__main__":
    main()
