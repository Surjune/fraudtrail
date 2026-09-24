"""Counts written the way an investigator writes them: one card, three cards.

"card(s)" is how a template hedges, and it reaches a report a regulator reads. Every
counted noun in an evidence claim, an action reason or a narrative goes through here.
"""

from __future__ import annotations


def noun(n: int, singular: str, plural: str = "") -> str:
    """The noun as it reads after a count of n. Pass the plural where adding s is wrong."""
    return singular if n == 1 else (plural or f"{singular}s")


def counted(n: int, singular: str, plural: str = "") -> str:
    """'1 card', '3 cards'."""
    return f"{n} {noun(n, singular, plural)}"


def was_or_were(n: int) -> str:
    return "was" if n == 1 else "were"
