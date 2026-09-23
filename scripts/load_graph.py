"""Create the FraudTrail graph in TigerGraph and load the exported files.

Runs graph/schema.gsql, graph/vectors.gsql and graph/loading.gsql, then uploads every CSV
in data/processed/load to its file tag and prints the resulting counts.

Usage::

    uv run python scripts/export_graph.py     # writes the files first
    uv run python scripts/load_graph.py [--skip-schema] [--only transaction,made]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fraudtrail.config import load_settings
from fraudtrail.graph.client import GraphClient

log = logging.getLogger("load_graph")

REPO = Path(__file__).resolve().parent.parent
GRAPH_DIR = REPO / "graph"
JOB = "load_fraudtrail"

# Vertices before the edges that point at them.
LOAD_ORDER: tuple[str, ...] = (
    "customer",
    "card",
    "transaction",
    "device_profile",
    "email_domain",
    "billing_region",
    "pattern",
    "closed_case",
    "owns",
    "made",
    "from_device",
    "purchaser_email",
    "recipient_email",
    "billed_in",
    "next",
    "involves",
    "on_card",
    "connected_to",
    "case_pattern",
)


def files_for(load_dir: Path, name: str) -> list[Path]:
    """The single file for a whole export, or every monthly chunk, oldest first."""
    single = load_dir / f"{name}.csv"
    if single.exists():
        return [single]
    return sorted(load_dir.glob(f"{name}_*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skip-schema", action="store_true", help="only load data")
    parser.add_argument("--only", default="", help="comma-separated subset of load steps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    client = GraphClient.from_env(settings)
    load_dir = settings.load_dir

    if not args.skip_schema:
        for script in ("schema.gsql", "vectors.gsql", "loading.gsql"):
            client.run_file(GRAPH_DIR / script)

    wanted = [n.strip() for n in args.only.split(",") if n.strip()] or list(LOAD_ORDER)
    for name in wanted:
        if name not in LOAD_ORDER:
            raise SystemExit(f"unknown load step: {name}")
        paths = files_for(load_dir, name)
        if not paths:
            raise SystemExit(f"no export files for {name} in {load_dir}; run export_graph.py")
        for path in paths:
            size_mb = path.stat().st_size / 1_000_000
            log.info("loading %s (%.1f MB)", path.name, size_mb)
            client.upload(JOB, f"f_{name}", path)

    log.info("vertices: %s", client.vertex_counts())
    log.info("edges: %s", client.edge_counts())


if __name__ == "__main__":
    main()
