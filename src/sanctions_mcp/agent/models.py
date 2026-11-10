"""Chat model adapters for the triage agent.

Every adapter keeps the conversation in its provider's own message format, so provider
specific content (such as Claude's thinking blocks) goes back exactly as it was received.
The agent loop only sees `Turn` objects, and a failed provider call always surfaces as
ModelCallError, so the loop never needs to know any provider's exceptions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import anthropic
import httpx

from sanctions_mcp.settings import LLMSettings, SettingsError

# Server side refusal fallback on the Claude API. fallbacks="default" asks for the requested
# model's own default fallback configuration, and on a model without fallbacks the request
# simply runs on the requested model alone (both from the `fallbacks` docstring in the
# anthropic SDK 2.7.0, which also lists this beta). So it is always sent, and no list of
# model ids is kept here to go stale.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


class ModelCallError(RuntimeError):
    """A model call failed: an API or HTTP error, a timeout, or a reply that cannot be read."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict
    error: str | None = None  # set when the arguments could not be read; input is then {}


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end_turn"  # tool_use | end_turn | refusal | max_tokens | other
    input_tokens: int = 0  # token counts cover every model that worked on this turn
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    refusal_category: str | None = None
    served_by: str | None = None  # the model that produced the reply, when the API says
    fallback: bool = False  # True when a fallback model, not the requested one, served it
    # Tokens of models other than the requested one (a refusal fallback), per model. They
    # are part of the counts above, but their price is not in settings.
    other_model_tokens: dict[str, dict[str, int]] = field(default_factory=dict)


class ChatSession(Protocol):
    def send(self, text: str) -> Turn: ...

    def send_tool_results(self, results: Sequence[tuple[str, str, bool]]) -> Turn:
        """results: (tool_call_id, content, is_error) for every call of the last turn."""
        ...


class ChatModel(Protocol):
    name: str

    def start(self, system: str, tools: Sequence[dict]) -> ChatSession: ...


def same_model(a: str, b: str) -> bool:
    """Model ids on Bedrock carry an `anthropic.` prefix; it does not make another model."""
    return a.removeprefix("anthropic.") == b.removeprefix("anthropic.")


def _refuse_an_empty_bedrock_key() -> None:
    """The anthropic SDK treats AWS_BEARER_TOKEN_BEDROCK as set whenever it exists, even
    empty (the line copied from .env.example), and then sends that empty key in place of
    the AWS credentials, so every call would fail. Stop before any call instead."""
    key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if key is not None and not key.strip():
        raise SettingsError(
            "AWS_BEARER_TOKEN_BEDROCK is set but empty, so the anthropic SDK would send an "
            "empty Bedrock API key instead of using your AWS credentials. Fill it in, or "
            "remove it (delete its line from .env)."
        )


# ---------------------------------------------------------------- Anthropic and Bedrock


class AnthropicModel:
    """Claude through the official `anthropic` SDK, on the Claude API or Amazon Bedrock."""

    def __init__(self, settings: LLMSettings, client=None):
        self.settings = settings
        bedrock = settings.provider == "bedrock"
        if client is None:
            if bedrock:
                _refuse_an_empty_bedrock_key()
            try:
                client = (
                    anthropic.AnthropicBedrockMantle(aws_region=settings.aws_region)
                    if bedrock
                    else anthropic.Anthropic()
                )
            except anthropic.AnthropicError as err:
                msg = f"could not set up the {settings.provider} client: {err}"
                raise SettingsError(msg) from None
            if not bedrock and not any(
                getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials")
            ):
                raise SettingsError("provider anthropic needs ANTHROPIC_API_KEY in the environment")
        self.client = client
        model = settings.model
        if bedrock and not model.startswith("anthropic."):
            model = f"anthropic.{model}"
        self.model = model
        # Claude API: always ask for the server side refusal fallback (see FALLBACK_BETA).
        # Bedrock: no fallback. Bedrock does not offer the server side one, and the SDK's
        # client side fallback middleware needs an explicit chain of Bedrock model ids,
        # which this project does not guess. A refusal on Bedrock ends the alert as an
        # escalation, and the run summary records that fallbacks were off.
        self.use_fallbacks = not bedrock
        self.fallbacks = (
            "server side, default chain" if self.use_fallbacks else "off (not offered on Bedrock)"
        )
        self.name = f"{settings.provider}:{model}"

    def start(self, system: str, tools: Sequence[dict]) -> AnthropicSession:
        return AnthropicSession(self, system, tools)


class AnthropicSession:
    def __init__(self, owner: AnthropicModel, system: str, tools: Sequence[dict]):
        self.owner = owner
        # The system prompt and tools never change between alerts, so they are cached.
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = [{**t, "strict": True} for t in tools]
        self.messages: list[dict] = []

    def _call(self) -> Turn:
        s = self.owner.settings
        kwargs = {
            "model": self.owner.model,
            "max_tokens": s.max_tokens,
            "system": self.system,
            "tools": self.tools,
            "messages": self.messages,
            "output_config": {"effort": s.effort},
            # Automatic caching: a second breakpoint that moves to the end of the growing
            # alert history on every call, next to the fixed one on tools and system.
            "cache_control": {"type": "ephemeral"},
        }
        try:
            if self.owner.use_fallbacks:
                response = self.owner.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            else:
                response = self.owner.client.messages.create(**kwargs)
        except anthropic.AnthropicError as err:
            raise ModelCallError(f"{type(err).__name__}: {err}") from err
        # Append the full content, thinking blocks included, exactly as received.
        self.messages.append({"role": "assistant", "content": response.content})
        requested = self.owner.model
        served = getattr(response, "model", None) or requested
        usage = response.usage
        # usage.iterations has one entry per attempt when a fallback ran: declined attempts
        # are `message` entries and the attempt that answered is `fallback_message`. Later
        # turns that stay on the fallback model carry a `fallback_message` entry too.
        attempts = [
            i
            for i in (getattr(usage, "iterations", None) or [])
            if getattr(i, "type", None) in ("message", "fallback_message")
        ]
        fallback = any(i.type == "fallback_message" for i in attempts)
        per_model: dict[str, dict[str, int]] = {}

        def add(model: str, counts: Sequence[int | None]) -> None:
            slot = per_model.setdefault(model, dict.fromkeys(TOKEN_FIELDS, 0))
            for name, value in zip(TOKEN_FIELDS, counts, strict=True):
                slot[name] += value or 0

        if attempts:
            for i in attempts:
                default = served if i.type == "fallback_message" else requested
                add(getattr(i, "model", None) or default, _counts(i))
        else:
            add(served if fallback else requested, _counts(usage))
        totals = {name: sum(c[name] for c in per_model.values()) for name in TOKEN_FIELDS}
        details = getattr(response, "stop_details", None)
        return Turn(
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[
                ToolCall(b.id, b.name, dict(b.input))
                for b in response.content
                if b.type == "tool_use"
            ],
            stop=_stop(response.stop_reason),
            refusal_category=getattr(details, "category", None) if details else None,
            served_by=served,
            fallback=fallback,
            other_model_tokens={m: c for m, c in per_model.items() if not same_model(m, requested)},
            **totals,
        )

    def send(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: Sequence[tuple[str, str, bool]]) -> Turn:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": i, "content": c, "is_error": e}
                    for i, c, e in results
                ],
            }
        )
        return self._call()


def _counts(usage) -> tuple[int | None, ...]:
    return (
        getattr(usage, "input_tokens", 0),
        getattr(usage, "output_tokens", 0),
        getattr(usage, "cache_read_input_tokens", 0),
        getattr(usage, "cache_creation_input_tokens", 0),
    )


def _stop(reason: str | None) -> str:
    return reason if reason in ("tool_use", "end_turn", "refusal", "max_tokens") else "other"


# ---------------------------------------------------------------- OpenAI style APIs


class OpenAICompatibleModel:
    """OpenAI, Azure OpenAI or a local Ollama server, through the chat completions API."""

    fallbacks = "off (not offered by this provider)"

    def __init__(self, settings: LLMSettings, client=None):
        self.settings = settings
        headers: dict[str, str] = {}
        if settings.provider == "openai":
            base = settings.base_url or "https://api.openai.com/v1"
            headers["Authorization"] = f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"
        elif settings.provider == "azure":
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
            if not settings.base_url and not endpoint:
                raise SettingsError(
                    "provider azure needs AZURE_OPENAI_ENDPOINT (for example "
                    "https://NAME.openai.azure.com) or SANCTIONS_LLM_BASE_URL"
                )
            base = settings.base_url or endpoint + "/openai/v1"
            headers["api-key"] = os.environ.get("AZURE_OPENAI_API_KEY", "")
        else:
            base = settings.base_url or "http://localhost:11434/v1"
        self.base = base.rstrip("/")
        self.client = client or httpx.Client(timeout=120.0, headers=headers)
        self.name = f"{settings.provider}:{settings.model}"

    def start(self, system: str, tools: Sequence[dict]) -> OpenAISession:
        return OpenAISession(self, system, tools)


class OpenAISession:
    def __init__(self, owner: OpenAICompatibleModel, system: str, tools: Sequence[dict]):
        self.owner = owner
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def _call(self) -> Turn:
        payload = {
            "model": self.owner.settings.model,
            "messages": self.messages,
            "tools": self.tools,
            "max_completion_tokens": self.owner.settings.max_tokens,
        }
        try:
            response = self.owner.client.post(f"{self.owner.base}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as err:
            raise ModelCallError(f"{type(err).__name__}: {err}") from err
        self.messages.append(message)
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        tokens = {
            "input_tokens": (usage.get("prompt_tokens") or 0) - cached,
            "output_tokens": usage.get("completion_tokens") or 0,
            "cache_read_tokens": cached,
            # These APIs have no fallback, and they report a dated name for the same model.
            "served_by": self.owner.settings.model,
        }
        text = message.get("content") or ""
        # The finish reason comes first: a reply cut off at the token limit or stopped by a
        # content filter may hold half written tool calls, which are never run.
        finish = choice.get("finish_reason")
        if finish == "length":
            return Turn(text, [], "max_tokens", **tokens)
        if finish == "content_filter":
            return Turn(text, [], "refusal", **tokens)
        calls = [_read_call(c) for c in message.get("tool_calls") or []]
        return Turn(text, calls, "tool_use" if calls else "end_turn", **tokens)

    def send(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: Sequence[tuple[str, str, bool]]) -> Turn:
        for call_id, content, is_error in results:
            body = f"ERROR: {content}" if is_error else content
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": body})
        return self._call()


def _read_call(call: dict) -> ToolCall:
    """A tool call from a chat completions reply. Arguments that are not a JSON object are
    kept as a marked call with the same id, so the agent can answer it with an error and
    the transcript stays valid (every tool call id needs a reply)."""
    fn = call.get("function") or {}
    call_id, name, raw = str(call.get("id", "")), str(fn.get("name", "")), fn.get("arguments")
    if isinstance(raw, dict):
        return ToolCall(call_id, name, raw)
    try:
        args = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return ToolCall(call_id, name, {}, error="the arguments were not valid JSON")
    if not isinstance(args, dict):
        return ToolCall(call_id, name, {}, error="the arguments were not a JSON object")
    return ToolCall(call_id, name, args)


def model_from_settings(settings: LLMSettings) -> ChatModel:
    if settings.provider in ("anthropic", "bedrock"):
        return AnthropicModel(settings)
    return OpenAICompatibleModel(settings)
