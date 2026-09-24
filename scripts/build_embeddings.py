"""Embed the corpus and the closed cases into the graph's vector store.

Writes three things TigerVector can search:

* `DocChunk.content_emb` for the Fraud Policy, the documented patterns, the regulatory
  references and the answer format, so a decision can cite the bank's actual wording.
* `ClosedCase.notes_emb` for every analyst note, so retrieval finds cases that read like
  the one being investigated rather than ones sharing a keyword.

Run once after the graph is loaded, and again whenever the corpus changes.

Usage::

    uv run python scripts/build_embeddings.py [--only docs|cases] [--batch 200]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from fraudtrail.config import load_settings
from fraudtrail.graph.client import GraphClient
from fraudtrail.graphrag.corpus import read_corpus
from fraudtrail.graphrag.embed import LocalEmbedder

log = logging.getLogger("build_embeddings")

REPO = Path(__file__).resolve().parent.parent

# Enough rows per request to be quick, small enough that one failure is cheap to repeat.
DEFAULT_BATCH = 200

# Past the dataset's own dates: every closed case is embedded, and each retrieval applies
# the cut-off its own investigation is allowed to see.
ALL_CASES_UNTIL = "2030-01-01 00:00:00"
MAX_CASES = 8000


def _batches(rows: list[tuple[str, dict[str, Any]]], size: int) -> list[list[tuple[str, Any]]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def embed_docs(client: GraphClient, embedder: LocalEmbedder, batch: int, readme: Path) -> int:
    chunks = read_corpus(readme)
    log.info("corpus: %d chunks from %s", len(chunks), readme.name)
    vectors = embedder.documents([c.content for c in chunks])
    rows: list[tuple[str, dict[str, Any]]] = [
        (
            chunk.chunk_id,
            {
                "source": chunk.source,
                "section": chunk.section,
                "content": chunk.content,
                "content_emb": vector,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    written = 0
    for part in _batches(rows, batch):
        written += client.upsert_vertices("DocChunk", part)
    return written


def embed_closed_cases(
    client: GraphClient, embedder: LocalEmbedder, batch: int, offset: int, limit: int
) -> int:
    result = client.run_query(
        "closed_case_notes", {"before_ts": ALL_CASES_UNTIL, "max_cases": MAX_CASES}
    )
    cases: list[dict[str, Any]] = []
    for block in result:
        if isinstance(block, dict) and isinstance(block.get("cases"), list):
            cases.extend(v for v in block["cases"] if isinstance(v, dict))
    cases.sort(key=lambda c: str(c.get("v_id", "")))
    # A stable order makes a slice mean the same thing on every run, so a long job can
    # be finished in pieces after an interruption.
    selected = cases[offset : offset + limit] if limit else cases[offset:]
    log.info("closed cases: %d, embedding %d from offset %d", len(cases), len(selected), offset)

    written = 0
    for part in _batches([(str(c["v_id"]), c.get("attributes", {})) for c in selected], batch):
        # A case with no note has nothing to embed, and an empty vector would still be
        # returned by a search as if it meant something.
        notes = [(case_id, str(attrs.get("analyst_notes") or "")) for case_id, attrs in part]
        usable = [(case_id, note) for case_id, note in notes if note.strip()]
        if not usable:
            continue
        vectors = embedder.documents([note for _, note in usable])
        rows: list[tuple[str, dict[str, Any]]] = [
            (case_id, {"notes_emb": vector})
            for (case_id, _), vector in zip(usable, vectors, strict=True)
        ]
        written += client.upsert_vertices("ClosedCase", rows)
        log.info("  embedded %d/%d of this slice", written, len(selected))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=("docs", "cases"), default="")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--offset", type=int, default=0, help="skip this many cases")
    parser.add_argument("--limit", type=int, default=0, help="0 means all of them")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_settings()
    client = GraphClient.from_env(settings)
    embedder = LocalEmbedder(settings.embedding_model)

    if args.only != "cases":
        log.info(
            "documents embedded: %d",
            embed_docs(client, embedder, args.batch, REPO / "data" / "raw" / "README.md"),
        )
    if args.only != "docs":
        log.info(
            "closed cases embedded: %d",
            embed_closed_cases(client, embedder, args.batch, args.offset, args.limit),
        )


if __name__ == "__main__":
    main()
