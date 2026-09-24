import json

import pytest

from fraudtrail.llm.client import (
    Completion,
    FallbackClient,
    LlmClient,
    LlmError,
    QuotaExhausted,
    is_daily_quota,
    retry_delay_s,
)


def refusal(quota_id: str, retry: str = "20s") -> str:
    """The shape of Gemini's 429 body, trimmed to the parts the client reads."""
    return json.dumps(
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": quota_id}],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry},
                ],
            }
        }
    )


def test_a_spent_daily_quota_is_recognised() -> None:
    assert is_daily_quota(refusal("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))


def test_a_busy_minute_is_not_a_spent_day() -> None:
    assert not is_daily_quota(refusal("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"))


def test_a_body_that_is_not_json_is_not_a_daily_quota() -> None:
    assert not is_daily_quota("<html>Too Many Requests</html>")


def test_the_requested_wait_is_honoured_and_capped() -> None:
    assert retry_delay_s(refusal("PerMinute", "20s")) == 20.0
    assert retry_delay_s(refusal("PerMinute", "3600s")) == 60.0
    assert retry_delay_s("not json") == 0.0


class Spent:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, prompt: str) -> Completion:
        self.calls += 1
        raise QuotaExhausted("daily quota spent")


class Answers:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def complete(self, system: str, prompt: str) -> Completion:
        self.calls += 1
        return Completion(self._text, 10)


class BusyOnce:
    """Overloaded for the first request, fine afterwards."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, prompt: str) -> Completion:
        self.calls += 1
        if self.calls == 1:
            raise LlmError("HTTP 503: overloaded")
        return Completion("from the first", 10)


class Busy:
    def complete(self, system: str, prompt: str) -> Completion:
        raise LlmError("HTTP 503: overloaded")


def test_the_next_model_takes_over_when_the_first_is_spent() -> None:
    spent, backup = Spent(), Answers("from the backup")
    client = FallbackClient([("first", spent), ("second", backup)])
    assert client.complete("s", "p").text == "from the backup"
    assert client.complete("s", "p").text == "from the backup"
    # A spent model is not asked again: its quota does not come back inside the run.
    assert spent.calls == 1
    assert backup.calls == 2


def test_every_model_spent_is_reported_as_a_spent_quota() -> None:
    clients: list[tuple[str, LlmClient]] = [("first", Spent()), ("second", Spent())]
    with pytest.raises(QuotaExhausted):
        FallbackClient(clients).complete("s", "p")


def test_a_busy_model_hands_one_request_on_and_is_asked_first_next_time() -> None:
    first, backup = BusyOnce(), Answers("from the backup")
    client = FallbackClient([("first", first), ("second", backup)])
    assert client.complete("s", "p").text == "from the backup"
    # Load passes; only a spent quota drops a model for the rest of the run.
    assert client.complete("s", "p").text == "from the first"
    assert backup.calls == 1


def test_every_model_busy_is_not_reported_as_a_spent_quota() -> None:
    client = FallbackClient([("first", Busy()), ("second", Busy())])
    with pytest.raises(LlmError) as raised:
        client.complete("s", "p")
    # The narrator gives up on the model for good only on a spent quota; load is not that.
    assert not isinstance(raised.value, QuotaExhausted)


def test_a_spent_model_behind_a_busy_one_is_dropped() -> None:
    spent = Spent()
    client = FallbackClient([("first", Busy()), ("second", spent), ("third", Answers("ok"))])
    assert client.complete("s", "p").text == "ok"
    assert client.complete("s", "p").text == "ok"
    assert spent.calls == 1
