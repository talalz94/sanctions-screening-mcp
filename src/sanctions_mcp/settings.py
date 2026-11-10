"""Settings for the triage agent: which model provider, model and effort to use.

Defaults live in settings.toml (committed, no secrets). Any field can be overridden with an
environment variable, so a run can switch provider or model without touching code. This
docstring lists every environment variable the project reads, and .env.example lists the
same names with empty values. Nothing loads a .env file by itself: export the variables in
your shell, or run `uv run --env-file .env sanctions ...`. A SANCTIONS_ variable set to an
empty value counts as unset.

Settings overrides, read by `load_settings`:

    SANCTIONS_LLM_PROVIDER      anthropic | bedrock | openai | azure | ollama
    SANCTIONS_LLM_MODEL         e.g. claude-opus-5-5
    SANCTIONS_LLM_EFFORT        low | medium | high | xhigh | max (Claude models)
    SANCTIONS_LLM_MAX_TOKENS    1 to 16000, per model call (calls are not streamed)
    SANCTIONS_PRICE_INPUT       USD per 1M input tokens of the configured model
    SANCTIONS_PRICE_OUTPUT      USD per 1M output tokens of the configured model
    SANCTIONS_PRICE_CACHE_READ  USD per 1M cache read tokens of the configured model
    SANCTIONS_AWS_REGION        provider bedrock only, default us-east-1
    SANCTIONS_LLM_BASE_URL      replaces the OpenAI, Azure or Ollama endpoint

Data location, read by `sanctions_mcp.data`:

    SANCTIONS_DATA_DIR          where the list, eval sets and models live (default ./data)

Credentials. They are never read from settings.toml, and this project never prints them:

    ANTHROPIC_API_KEY           provider anthropic, read by the anthropic SDK
    AWS_PROFILE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
                                provider bedrock, read by the AWS SDK through the anthropic
                                SDK (any other source in the usual AWS chain works too)
    AWS_BEARER_TOKEN_BEDROCK    provider bedrock, a Bedrock API key used instead of the
                                AWS credentials above. The SDK uses it whenever it is set,
                                even empty, so an empty one is refused before any call.
    OPENAI_API_KEY              provider openai, read by this project's own adapter
    AZURE_OPENAI_API_KEY        provider azure, read by this project's own adapter
    AZURE_OPENAI_ENDPOINT       provider azure, e.g. https://NAME.openai.azure.com; not
                                needed when SANCTIONS_LLM_BASE_URL is set
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path

PROVIDERS = ("anthropic", "bedrock", "openai", "azure", "ollama")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# The agent's model calls are not streamed. The anthropic SDK refuses an unstreamed call
# whose max_tokens could outlast its ten minute timeout, so the setting stays below that.
MAX_TOKENS_LIMIT = 16000


class SettingsError(ValueError):
    """A setting that cannot work, with a message that says how to fix it."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000
    # List prices of the configured model, used to report cost per 1,000 alerts. Check them
    # against the provider's current price list before quoting a cost. Turns answered by a
    # different model (a refusal fallback) are reported as unpriced, never at these rates.
    price_input: float = 4.0
    price_output: float = 20.0
    price_cache_read: float = 0.20
    aws_region: str = "us-east-1"
    base_url: str = ""

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise SettingsError(f"provider {self.provider!r} is not one of {PROVIDERS}")
        if self.effort not in EFFORTS:
            raise SettingsError(f"effort {self.effort!r} is not one of {EFFORTS}")
        if not 1 <= self.max_tokens <= MAX_TOKENS_LIMIT:
            raise SettingsError(
                f"max_tokens is {self.max_tokens}, but it must be between 1 and "
                f"{MAX_TOKENS_LIMIT}: the agent does not stream its model calls, and longer "
                "unstreamed calls fail. Lower max_tokens in settings.toml or "
                "SANCTIONS_LLM_MAX_TOKENS."
            )

    def cost_usd(self, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> float:
        return (
            input_tokens * self.price_input
            + output_tokens * self.price_output
            + cache_read_tokens * self.price_cache_read
        ) / 1_000_000

    def cache_write_usd(self, tokens: int) -> float:
        """Cache writes (5 minute lifetime) cost 1.25 times the input price."""
        return tokens * self.price_input * 1.25 / 1_000_000


_ENV = {
    "provider": "SANCTIONS_LLM_PROVIDER",
    "model": "SANCTIONS_LLM_MODEL",
    "effort": "SANCTIONS_LLM_EFFORT",
    "max_tokens": "SANCTIONS_LLM_MAX_TOKENS",
    "price_input": "SANCTIONS_PRICE_INPUT",
    "price_output": "SANCTIONS_PRICE_OUTPUT",
    "price_cache_read": "SANCTIONS_PRICE_CACHE_READ",
    "aws_region": "SANCTIONS_AWS_REGION",
    "base_url": "SANCTIONS_LLM_BASE_URL",
}


def load_settings(path: Path | None = None, env: Mapping[str, str] | None = None) -> LLMSettings:
    """Settings from a file's [llm] table, then the environment. With no path, settings.toml
    in the working folder is read if it is there. A path that was given must exist, and the
    file must hold an [llm] table with only known keys, so a typo never quietly falls back
    to the default model."""
    env = os.environ if env is None else env
    values: dict = {}
    if path is None:
        path = Path("settings.toml") if Path("settings.toml").exists() else None
    elif not path.exists():
        raise SettingsError(f"settings file {path.as_posix()} not found")
    types = {f.name: f.type for f in fields(LLMSettings)}
    if path is not None:
        try:
            table = tomllib.loads(path.read_text(encoding="utf-8")).get("llm")
        except tomllib.TOMLDecodeError as err:
            raise SettingsError(f"{path.as_posix()} is not valid TOML: {err}") from None
        if not isinstance(table, dict):
            raise SettingsError(f"{path.as_posix()} has no [llm] table")
        unknown = sorted(set(table) - set(types))
        if unknown:
            raise SettingsError(f"{path.as_posix()}: unknown setting {', '.join(unknown)}")
        values.update(table)
    for name, var in _ENV.items():
        if env.get(var):
            values[name] = env[var]
    cast = {"int": int, "float": float, "str": str}
    clean = {}
    for k, v in values.items():
        if k not in types:
            continue
        try:
            clean[k] = cast[str(types[k])](v)
        except ValueError:
            raise SettingsError(f"setting {k} must be a number") from None
    return replace(LLMSettings(), **clean)
