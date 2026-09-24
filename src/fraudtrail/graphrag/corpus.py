"""The documents the agent can cite, cut into retrievable pieces.

Everything here comes from the round's own material: the Fraud Policy, the five
documented patterns, the regulatory references and the rules, all of which ship with the
dataset. Nothing is paraphrased and nothing is invented, because a chunk retrieved to
justify a decision has to be the bank's actual wording.

Chunks are cut at paragraph boundaries under their own heading, so a retrieved piece
reads as a whole thought and carries the section it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# The embedding model reads about 512 tokens; this keeps a chunk inside that without
# splitting mid-sentence.
MAX_CHUNK_CHARS = 1400

# Below this a chunk is a heading or a stub, and retrieving it tells nobody anything.
MIN_CHUNK_CHARS = 120

# Everything after this heading in the dataset README is the policy itself.
POLICY_HEADING = "# Fraud Policy"

SOURCE_POLICY = "fraud_policy"
SOURCE_GUIDE = "dataset_guide"

# Sections that are instructions to the team rather than material an investigation can
# cite. Retrieving the exam's own case table or the answer-format example to justify a
# decision would be worse than retrieving nothing.
SKIPPED_SECTIONS = frozenset(
    {
        "start here: your first two hours",
        "files in this folder",
        "the 20 cases",
        "the case pack",
        "example",
        "attribution",
        "suggested graph schema",
    }
)

SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    section: str
    content: str


def _slug(text: str) -> str:
    return SLUG.sub("-", text.lower()).strip("-")[:48]


def _paragraphs(text: str) -> list[str]:
    """Paragraphs, with anything longer than a chunk broken at its line ends.

    A markdown table or a long list is one paragraph with no blank line in it, and
    embedding only its first part would silently lose the rest.
    """
    out: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if len(paragraph) <= MAX_CHUNK_CHARS:
            out.append(paragraph)
            continue
        current = ""
        for line in paragraph.splitlines():
            if current and len(current) + len(line) + 1 > MAX_CHUNK_CHARS:
                out.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            out.append(current)
    return out


def _split(text: str) -> list[str]:
    """Paragraphs joined up to the size limit, never cut mid-paragraph."""
    pieces: list[str] = []
    current = ""
    for paragraph in _paragraphs(text):
        if current and len(current) + len(paragraph) + 2 > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def read_corpus(readme: Path) -> tuple[Chunk, ...]:
    """Every section of the dataset README, as chunks tagged with where they came from."""
    text = readme.read_text(encoding="utf-8")
    policy_at = text.find(POLICY_HEADING)
    sections: list[tuple[str, str, str]] = []

    for source, body in (
        (SOURCE_GUIDE, text[:policy_at] if policy_at > 0 else text),
        (SOURCE_POLICY, text[policy_at:] if policy_at > 0 else ""),
    ):
        heading = ""
        buffer: list[str] = []
        for line in body.splitlines():
            if line.startswith("#"):
                if heading and buffer:
                    sections.append((source, heading, "\n".join(buffer)))
                heading = line.lstrip("#").strip()
                buffer = []
            else:
                buffer.append(line)
        if heading and buffer:
            sections.append((source, heading, "\n".join(buffer)))

    chunks: list[Chunk] = []
    for source, section, body in sections:
        if section.lower() in SKIPPED_SECTIONS:
            continue
        for index, piece in enumerate(_split(body), start=1):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{source}:{_slug(section)}:{index:02d}",
                    source=source,
                    section=section,
                    # The heading is part of what the chunk means, so it is embedded too.
                    content=f"{section}\n\n{piece}",
                )
            )
    return tuple(chunks)
