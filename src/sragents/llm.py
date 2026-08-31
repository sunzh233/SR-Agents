"""Minimal OpenAI-compatible LLM wrapper.

Works with vLLM (OpenAI-compatible API), OpenAI, or any compatible endpoint.
Config via CLI args (--model, --api-base) and env vars (OPENAI_API_BASE, OPENAI_API_KEY).
"""

import os
import re
from dataclasses import dataclass

from openai import OpenAI


def create_llm_client(
    api_base: str | None = None, api_key: str | None = None
) -> OpenAI:
    """Create OpenAI-compatible client (works with vLLM, OpenAI, etc.)."""
    base_url = api_base or os.environ.get("OPENAI_API_BASE")
    key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    # Large concurrent runs can queue behind one model server for longer than
    # the SDK default; use one shared explicit timeout for all callers.
    return OpenAI(base_url=base_url, api_key=key, timeout=3600)


_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from an LLM response.

    Handles both well-formed `<think>...</think>answer` and truncation
    cases where generation is cut mid-thinking (unclosed tag).
    """
    if "<think>" not in text:
        return text
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.lstrip()


def get_extra_body(model: str, thinking: bool = False) -> dict | None:
    """Return per-model extra_body for thinking/reasoning control.

    By default (thinking=False) suppresses thinking on hybrid-thinking
    models so they're comparable to non-reasoning baselines. GPT-5 is a
    pure reasoning model; we always run it at minimal effort.

      - Qwen3: chat_template_kwargs.enable_thinking
      - GLM-5 / Kimi: enable_thinking
      - GPT-5: reasoning_effort="minimal" (always)
      - Others (Llama, Mistral, MiniMax, ...): no flag
    """
    basename = model.lower().rsplit("/", 1)[-1]

    if "qwen3" in basename:
        return {"chat_template_kwargs": {"enable_thinking": thinking}}
    if "gpt-5" in basename:
        return {"reasoning_effort": "minimal"}
    if "glm-5" in basename or "kimi" in basename:
        return {"enable_thinking": thinking}
    return None


@dataclass(frozen=True)
class ChatResult:
    content: str
    finish_reason: str
    completion_tokens: int | None


def chat_with_metadata(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    extra_body: dict | None = None,
) -> ChatResult:
    """Send one chat request and retain the length-control metadata."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if stop:
        kwargs["stop"] = stop
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    return ChatResult(
        content=choice.message.content or "",
        finish_reason=getattr(choice, "finish_reason", None) or "stop",
        completion_tokens=(
            int(completion_tokens) if completion_tokens is not None else None
        ),
    )


def chat(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    extra_body: dict | None = None,
) -> str:
    """Send chat completion request, return content string."""
    return chat_with_metadata(
        client, model, prompt, system=system, temperature=temperature,
        max_tokens=max_tokens, stop=stop, extra_body=extra_body,
    ).content


def chat_messages(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    extra_body: dict | None = None,
) -> str:
    """Send chat completion with an explicit messages list (for multi-turn)."""
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if stop:
        kwargs["stop"] = stop
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
