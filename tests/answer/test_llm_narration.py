from typing import cast

from fraudtrail.answer.llm_narration import LlmNarrator
from fraudtrail.answer.narration import Narration, Narrator
from fraudtrail.llm.client import Completion, LlmError

SOURCE_SUMMARY = (
    "The bank's model scored a $74.96 card-not-present purchase online on card "
    "C13487-K1 at 0.87. "
    "Twenty cards share the device profile in the thirty days before the alert. "
    "Assessed as fraud at 0.91, with 1,248.30 USD exposure."
)

SOURCE_NARRATIVE = (
    "Between 2016-11-14 and 2016-11-22, card C13487-K1 belonging to customer C13487 was "
    "used for 6 transactions totalling $1,248.30 that the bank believes were "
    "unauthorised. "
    "The activity ran through online channels under product code C. "
    "Every transaction came from one device profile. "
    "That device profile reaches 20 cards across 20 customers. "
    "The cardholder was asked to validate the activity and denied it. "
    "The total suspicious amount is $1,248.30, and the recommended actions are "
    "block_card, create_case."
)

NARRATION = cast(Narration, object())


class StubFallback:
    """Stands in for TemplateNarrator: the facts come from here, never from the model."""

    def summary(self, n: Narration) -> str:
        return SOURCE_SUMMARY

    def sar_narrative(self, n: Narration) -> str:
        return SOURCE_NARRATIVE


class StubClient:
    def __init__(self, reply: str, tokens: int = 100) -> None:
        self._reply = reply
        self.tokens = tokens

    def complete(self, system: str, prompt: str) -> Completion:
        return Completion(self._reply, self.tokens)


class FailingClient:
    def complete(self, system: str, prompt: str) -> Completion:
        raise LlmError("rate limited")


def narrator(reply: str) -> LlmNarrator:
    return LlmNarrator(StubClient(reply), fallback=cast(Narrator, StubFallback()))


def test_faithful_rewrite_is_kept() -> None:
    rewritten = (
        "Between 2016-11-14 and 2016-11-22, six transactions totalling $1,248.30 on card "
        "C13487-K1, held by customer C13487, are believed to be unauthorised. "
        "All ran online under product code C. "
        "Every one came from a single device profile. "
        "That profile reaches 20 cards across 20 customers. "
        "The cardholder denied the activity when asked to validate it. "
        "The suspicious total is $1,248.30 and the recommended actions are block_card "
        "and create_case."
    )
    n = narrator(rewritten)
    assert n.sar_narrative(NARRATION) == rewritten
    assert n.rejected == 0
    assert n.tokens == 100


def test_invented_amount_is_rejected() -> None:
    n = narrator(SOURCE_NARRATIVE.replace("$1,248.30 that", "$9,999.00 that"))
    assert n.sar_narrative(NARRATION) == SOURCE_NARRATIVE
    assert n.rejected == 1


def test_invented_identifier_is_rejected() -> None:
    n = narrator(SOURCE_NARRATIVE + " The activity also touched card C99999-K4.")
    assert n.sar_narrative(NARRATION) == SOURCE_NARRATIVE
    assert n.rejected == 1


def test_invented_date_is_rejected() -> None:
    n = narrator(SOURCE_NARRATIVE.replace("2016-11-14", "2016-10-01"))
    assert n.sar_narrative(NARRATION) == SOURCE_NARRATIVE
    assert n.rejected == 1


def test_dropped_identifier_is_rejected() -> None:
    n = narrator(
        SOURCE_NARRATIVE.replace(
            "card C13487-K1 belonging to customer C13487", "the card and its holder"
        )
    )
    assert n.sar_narrative(NARRATION) == SOURCE_NARRATIVE
    assert n.rejected == 1


def test_too_few_sentences_is_rejected() -> None:
    n = narrator(
        "Card C13487-K1 of customer C13487 saw 6 transactions totalling $1,248.30 "
        "between 2016-11-14 and 2016-11-22 from one device profile reaching 20 cards, "
        "which the cardholder denied, so the actions are block_card and create_case."
    )
    assert n.sar_narrative(NARRATION) == SOURCE_NARRATIVE
    assert n.rejected == 1


def test_code_fence_is_stripped() -> None:
    n = narrator("```\n" + SOURCE_NARRATIVE + "\n```")
    assert n.sar_narrative(NARRATION) == SOURCE_NARRATIVE
    assert n.rejected == 0


def test_summary_respects_its_own_sentence_bounds() -> None:
    rewritten = (
        "The bank's model scored a $74.96 online card-not-present purchase on card "
        "C13487-K1 at 0.87. "
        "Twenty cards share that device profile in the thirty days before the alert. "
        "The case is assessed as fraud at 0.91 with 1,248.30 USD of exposure."
    )
    n = narrator(rewritten)
    assert n.summary(NARRATION) == rewritten


def test_unreachable_model_falls_back_without_raising() -> None:
    n = LlmNarrator(FailingClient(), fallback=cast(Narrator, StubFallback()))
    assert n.sar_narrative(NARRATION) == SOURCE_NARRATIVE
    assert n.summary(NARRATION) == SOURCE_SUMMARY
    assert n.calls == 0
    assert n.rejected == 2
