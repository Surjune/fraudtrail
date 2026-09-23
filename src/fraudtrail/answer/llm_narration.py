"""The written parts of an answer, rewritten by a model under a fact check.

The model is given the templated text and asked to say the same thing better. It never
sees a decision to make and never supplies a fact: every identifier, amount and date in
what it returns must already appear in the text it was given. Anything else is a
fabrication in a document a regulator reads, so the answer falls back to the template and
says so in the log.

That check is the reason a model is safe to use here at all. Prose is the only thing it
can change, and prose is the only thing it is scored on.
"""

from __future__ import annotations

import logging
import re

from fraudtrail.answer.narration import Narration, Narrator, TemplateNarrator
from fraudtrail.answer.validate import count_sentences
from fraudtrail.llm.client import LlmClient, LlmError

log = logging.getLogger(__name__)

# Answer Format, Part 1: the case summary is "Two to six sentences".
SUMMARY_MIN_SENTENCES = 2
SUMMARY_MAX_SENTENCES = 6

# Answer Format, Part 2: the SAR narrative is "Six to twelve sentences".
NARRATIVE_MIN_SENTENCES = 6
NARRATIVE_MAX_SENTENCES = 12

# Case, card, customer and transaction identifiers, and the bank's closed-case numbers.
ID_TOKEN = re.compile(r"\b(?:[A-Z]{2,}-\d+|C\d+-K\d+|C\d{3,}|\d{5,})\b")
AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# A model asked for plain text sometimes wraps it in a code fence anyway.
FENCE = re.compile(r"^```[a-z]*\n|\n```$")

# Two failures in a row mean the key, the quota or the network is gone rather than one
# request being unlucky, and every further attempt pays the full retry backoff.
FAILURES_BEFORE_GIVING_UP = 2

SYSTEM = (
    "You write for a bank's fraud investigations team. The passage you are given was "
    "assembled from templates: the facts in it are correct and complete, but it reads "
    "mechanically. Your job is to write it the way an experienced investigator would.\n\n"
    "Rules, in order of importance:\n"
    "1. Never add a fact. Every identifier, amount, date, count and name in your answer "
    "must appear in the passage you were given. If something is not there, it does not "
    "exist. You have no other knowledge of this case.\n"
    "2. Never remove a fact. Every identifier, amount and date survives the rewrite.\n"
    "3. Never soften or strengthen a judgement. If the passage says the evidence is "
    "mixed, so does your answer.\n"
    "4. Do rewrite. The passage is not already good English: fix the machine artefacts "
    "such as 'transaction(s)', 'channel(s)' and 'product code(s)', join the clipped "
    "sentences into connected ones, and order the facts so the sequence of events is "
    "clear. Returning the passage unchanged is a failure.\n"
    "5. Write plain declarative prose. No headings, no bullet points, no code fences, no "
    "preamble such as 'Here is'. Return only the rewritten passage."
)


def _facts(text: str) -> tuple[set[str], set[str], set[str]]:
    ids = set(ID_TOKEN.findall(text))
    amounts = {a.replace(" ", "").replace(",", "") for a in AMOUNT.findall(text)}
    dates = set(DATE.findall(text))
    return ids, amounts, dates


def _unsupported(source: str, candidate: str) -> str:
    """Why the candidate cannot be trusted, or an empty string when it can."""
    source_ids, source_amounts, source_dates = _facts(source)
    ids, amounts, dates = _facts(candidate)
    invented_ids = sorted(ids - source_ids)
    invented_amounts = sorted(amounts - source_amounts)
    invented_dates = sorted(dates - source_dates)
    if invented_ids:
        return f"identifiers not in the evidence: {', '.join(invented_ids)}"
    if invented_amounts:
        return f"amounts not in the evidence: {', '.join(invented_amounts)}"
    if invented_dates:
        return f"dates not in the evidence: {', '.join(invented_dates)}"
    missing_ids = sorted(source_ids - ids)
    if missing_ids:
        return f"identifiers dropped from the evidence: {', '.join(missing_ids)}"
    return ""


def _clean(text: str) -> str:
    return FENCE.sub("", text.strip()).strip()


class LlmNarrator:
    """Wraps a model around `TemplateNarrator`, which stays the source of the facts."""

    def __init__(self, client: LlmClient, fallback: Narrator | None = None) -> None:
        self._client = client
        self._fallback = fallback or TemplateNarrator()
        self.calls = 0
        self.tokens = 0
        self.rejected = 0
        self._failures = 0
        self._given_up = False

    def summary(self, n: Narration) -> str:
        source = self._fallback.summary(n)
        prompt = (
            "Rewrite this case summary for the analyst who picks the case up next: what "
            "was flagged, what the evidence shows, and what was decided. "
            f"Use between {SUMMARY_MIN_SENTENCES} and {SUMMARY_MAX_SENTENCES} "
            "sentences.\n\n"
            f"{source}"
        )
        return self._rewrite(source, prompt, SUMMARY_MIN_SENTENCES, SUMMARY_MAX_SENTENCES)

    def sar_narrative(self, n: Narration) -> str:
        source = self._fallback.sar_narrative(n)
        prompt = (
            "Rewrite this suspicious activity report narrative so it stands on its own "
            "for a regulator who has not seen the case. Tell it in order: what was used, "
            "when, through which channel, what made it suspicious, and what the bank did. "
            "Keep every identifier, amount and date. "
            f"Use between {NARRATIVE_MIN_SENTENCES} and {NARRATIVE_MAX_SENTENCES} "
            "sentences.\n\n"
            f"{source}"
        )
        return self._rewrite(source, prompt, NARRATIVE_MIN_SENTENCES, NARRATIVE_MAX_SENTENCES)

    def take_tokens(self) -> int:
        """Tokens spent since the last call, so each answer records its own cost."""
        spent = self.tokens
        self.tokens = 0
        return spent

    def _rewrite(self, source: str, prompt: str, low: int, high: int) -> str:
        if self._given_up:
            return source
        try:
            completion = self._client.complete(SYSTEM, prompt)
        except LlmError as exc:
            self.rejected += 1
            self._failures += 1
            if self._failures >= FAILURES_BEFORE_GIVING_UP:
                self._given_up = True
                # A daily quota does not come back inside one run, and each attempt costs
                # the full backoff. Finish on templates instead of stalling every case.
                log.warning("model unreachable twice; writing the rest from templates")
            else:
                log.warning("model unavailable, keeping the templated text: %s", exc)
            return source
        self._failures = 0
        self.calls += 1
        self.tokens += completion.tokens

        candidate = _clean(completion.text)
        reason = ""
        if not candidate:
            reason = "the model returned nothing"
        elif not low <= count_sentences(candidate) <= high:
            reason = f"{count_sentences(candidate)} sentences, outside {low} to {high}"
        else:
            reason = _unsupported(source, candidate)
        if reason:
            self.rejected += 1
            log.warning("keeping the templated text: %s", reason)
            return source
        return candidate
