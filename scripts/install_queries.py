"""Create and install the GSQL queries the agent uses as tools.

Usage::

    uv run python scripts/install_queries.py [--no-install]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fraudtrail.config import load_settings
from fraudtrail.graph.client import GraphClient, GraphError

log = logging.getLogger("install_queries")

REPO = Path(__file__).resolve().parent.parent
QUERY_FILES = ("queries.gsql", "queries_memory.gsql")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-install", action="store_true", help="create but do not install")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    client = GraphClient.from_env(settings)
    for name in QUERY_FILES:
        client.run_file(REPO / "graph" / name)
    if not args.no_install:
        log.info("installing queries (this takes a few minutes)")
        install = f"""USE GRAPH {settings.tigergraph.graph}
INSTALL QUERY ALL"""
        try:
            client.gsql(install)
        except GraphError as exc:
            # Nothing to install is not a failure: re-running the script stays safe.
            if "installed already" not in str(exc):
                raise
    log.info("done")


if __name__ == "__main__":
    main()
