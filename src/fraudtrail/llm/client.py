"""One call to a hosted model, over the provider's HTTP API.

Each provider speaks JSON over HTTPS, so there is no SDK here: three small request
shapes are easier to read, pin and debug than three dependencies, and they cannot break
on an upgrade the night before a deadline.

A free-tier key is rate limited by the minute, so a refusal is ordinary rather than
exceptional and is retried with a widening gap. Anything still unusable after that raises
`LlmError`; the caller falls back to templated prose rather than shipping nothing.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from fraudtrail.config import ConfigError, LlmProvider, LlmSettings

log = logging.getLogger(__name__)

# Free-tier quotas are per minute, so the waits are seconds rather than milliseconds.
MAX_ATTEMPTS = 4
FIRST_BACKOFF_S = 2.0
BACKOFF_FACTOR = 3.0
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
TIMEOUT_S = 60.0

# The longest text asked for is a twelve-sentence narrative, but a reasoning model
# spends output tokens thinking before it writes: one rewrite here used 2,153 of them and
# returned a truncated fragment under a 900 budget. The ceiling covers both.
MAX_OUTPUT_TOKENS = 4000

# Determinism: the same evidence must produce the same case file on a re-run.
TEMPERATURE = 0.0

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENAI_BASE_URL = "https://api.openai.com/v1"

JsonDict = dict[str, Any]


class LlmError(RuntimeError):
    """The model could not be reached, or answered with something unusable."""


@dataclass(frozen=True)
class Completion:
    text: str
    tokens: int


class LlmClient(Protocol):
    def complete(self, system: str, prompt: str) -> Completion: ...


def _post(url: str, headers: dict[str, str], payload: JsonDict) -> JsonDict:
    """POST JSON and return the decoded body, retrying the statuses worth retrying."""
    body = json.dumps(payload).encode("utf-8")
    wait = FIRST_BACKOFF_S
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            if not isinstance(decoded, dict):
                raise LlmError(f"expected a JSON object, got {type(decoded).__name__}")
            return decoded
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code not in RETRY_STATUS:
                raise LlmError(last_error) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < MAX_ATTEMPTS:
            log.warning("model call failed (%s); retrying in %.0fs", last_error, wait)
            time.sleep(wait)
            wait *= BACKOFF_FACTOR
    raise LlmError(f"no usable answer after {MAX_ATTEMPTS} attempts. {last_error}")


def _require(value: str, field: str) -> str:
    if not value:
        raise ConfigError(f"{field} is required for this provider; see .env.example")
    return value


class AnthropicClient:
    def __init__(self, model: str, api_key: str) -> None:
        self._model = _require(model, "FRAUDTRAIL_LLM_MODEL")
        self._key = api_key

    def complete(self, system: str, prompt: str) -> Completion:
        data = _post(
            ANTHROPIC_URL,
            {
                "x-api-key": self._key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            {
                "model": self._model,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise LlmError(f"no content in the answer: {json.dumps(data)[:200]}")
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = data.get("usage", {})
        tokens = 0
        if isinstance(usage, dict):
            tokens = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        return Completion(text.strip(), tokens)


class GeminiClient:
    def __init__(self, model: str, api_key: str) -> None:
        self._model = _require(model, "FRAUDTRAIL_LLM_MODEL")
        self._key = api_key

    def complete(self, system: str, prompt: str) -> Completion:
        # The key goes in a header, never in the query string: a URL is logged and cached
        # in places a credential should never reach.
        data = _post(
            GEMINI_URL.format(model=self._model),
            {"x-goog-api-key": self._key, "content-type": "application/json"},
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": TEMPERATURE,
                    "maxOutputTokens": MAX_OUTPUT_TOKENS,
                },
            },
        )
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise LlmError(f"no candidates in the answer: {json.dumps(data)[:200]}")
        first = candidates[0]
        parts = first.get("content", {}).get("parts", []) if isinstance(first, dict) else []
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        usage = data.get("usageMetadata", {})
        tokens = int(usage.get("totalTokenCount", 0)) if isinstance(usage, dict) else 0
        return Completion(text.strip(), tokens)


class OpenAiCompatibleClient:
    """OpenAI's own API and every endpoint that copies it, which is most free tiers."""

    def __init__(self, model: str, api_key: str, base_url: str = "") -> None:
        self._model = _require(model, "FRAUDTRAIL_LLM_MODEL")
        self._key = api_key
        self._base = (base_url or OPENAI_BASE_URL).rstrip("/")

    def complete(self, system: str, prompt: str) -> Completion:
        data = _post(
            f"{self._base}/chat/completions",
            {"authorization": f"Bearer {self._key}", "content-type": "application/json"},
            {
                "model": self._model,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmError(f"no choices in the answer: {json.dumps(data)[:200]}")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        text = str(message.get("content", "")) if isinstance(message, dict) else ""
        usage = data.get("usage", {})
        tokens = int(usage.get("total_tokens", 0)) if isinstance(usage, dict) else 0
        return Completion(text.strip(), tokens)


def build_client(settings: LlmSettings) -> LlmClient | None:
    """The configured client, or None when no model is configured."""
    if not settings.enabled:
        return None
    if settings.provider is LlmProvider.ANTHROPIC:
        return AnthropicClient(settings.model, settings.api_key)
    if settings.provider is LlmProvider.GOOGLE:
        return GeminiClient(settings.model, settings.api_key)
    if settings.provider is LlmProvider.OPENAI:
        return OpenAiCompatibleClient(settings.model, settings.api_key, settings.base_url)
    return None
