"""Ranking closed cases by the analysts' own words.

Both providers retrieve the same memory the same way, so an offline run and a graph run
return the same prior cases. When the closed-case notes carry embeddings, the graph
answers this in one hop with `similar_closed_cases`; until then, overlap on distinctive
words over the same notes finds the same kind of case.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from fraudtrail.evidence.models import PriorCase

# Words that say nothing about which case is similar.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "this",
        "that",
        "they",
        "them",
        "their",
        "there",
        "here",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "it",
        "its",
        "as",
        "at",
        "by",
        "not",
        "no",
    ]
)

# Shorter tokens carry no signal, and a bare number is an identifier, not a word.
MIN_WORD_LENGTH = 4


def keywords(text: str) -> frozenset[str]:
    """Distinctive words, lowercased: the units both sides of a search share."""
    words = {token.strip(".,;:'\"()$").lower() for token in text.split()}
    return frozenset(
        w for w in words if len(w) >= MIN_WORD_LENGTH and w not in STOPWORDS and not w.isdigit()
    )


def most_similar(
    query: str,
    indexed: Sequence[tuple[PriorCase, frozenset[str]]],
    before: datetime,
    k: int,
) -> tuple[PriorCase, ...]:
    """The k closed cases sharing the most words with the query, newest first among equals.

    Cases opened at or after `before` are never returned: an investigation may only use
    what the bank knew when the alert fired.
    """
    words = keywords(query)
    if not words:
        return ()
    scored = [
        (len(words & note_words), case)
        for case, note_words in indexed
        if case.opened_at < before and words & note_words
    ]
    scored.sort(key=lambda item: (item[0], item[1].opened_at), reverse=True)
    return tuple(case for _, case in scored[:k])
