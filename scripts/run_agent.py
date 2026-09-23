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
from fraudtrail.answer.llm_narration import LlmNarrator
from fraudtrail.answer.narration import Narrator, TemplateNarrator
from fraudtrail.answer.validate import Level, check_answer
from fraudtrail.casepack import load_case_pack
from fraudtrail.config import Settings, load_settings
from fraudtrail.evidence.duckdb_provider import DuckDbProvider
from fraudtrail.evidence.provider import EvidenceProvider
from fraudtrail.evidence.tigergraph_provider import TigerGraphProvider
from fraudtrail.graph.memory import GraphMemory
from fraudtrail.investigate.runner import investigate
from fraudtrail.llm.client import build_client

log = logging.getLogger("run_agent")

REPO = Path(__file__).resolve().parent.parent


def build_provider(settings: Settings, offline: bool) -> EvidenceProvider:
    """TigerGraph is the runtime path; --offline runs the same investigation locally."""
    if offline:
        log.info("evidence from the local warehouse")
        return DuckDbProvider.open(settings.raw_dir, settings.processed_dir)
    log.info("evidence from TigerGraph at %s", settings.tigergraph.host)
    return TigerGraphProvider.from_env(settings)


def build_narrator(settings: Settings) -> Narrator:
    """The model when one is configured, templates when not. Both write every answer."""
    client = build_client(settings.llm)
    if client is None:
        log.info("no model configured: writing the prose from templates")
        return TemplateNarrator()
    log.info("prose from %s %s", settings.llm.provider.value, settings.llm.model)
    return LlmNarrator(client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", default="", help="comma-separated case IDs")
    parser.add_argument("--out", type=Path, default=REPO / "cases")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the local warehouse instead of TigerGraph (no workspace needed)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    cases = load_case_pack(settings.raw_dir / "case_pack.csv")
    wanted = {c.strip() for c in args.only.split(",") if c.strip()}
    if wanted:
        cases = tuple(case for case in cases if case.case_id in wanted)

    provider = build_provider(settings, offline=args.offline)
    narrator = build_narrator(settings)
    # The graph provider already holds an authenticated client; an offline run has none,
    # so its cases stay local and say so in the answer file.
    memory = GraphMemory(provider.client) if isinstance(provider, TigerGraphProvider) else None
    args.out.mkdir(parents=True, exist_ok=True)

    failures = 0
    for case in cases:
        result = investigate(provider, case)
        answer = build_answer(result, narrator)
        if isinstance(narrator, LlmNarrator):
            # The prose is written last, so the model's cost is known only now.
            answer = answer.model_copy(update={"tokens": narrator.take_tokens()})
        if memory is not None and memory.write(result, answer):
            stored = answer.case.model_copy(
                update={"written_to_graph": True, "graph_case_id": result.graph_case_id}
            )
            answer = answer.model_copy(update={"case": stored})
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
