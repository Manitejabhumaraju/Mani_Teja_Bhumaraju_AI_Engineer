"""
NVIDIA NIM client.

NIM is OpenAI-compatible - same wire format, same SDK, just a different
base_url and api_key. So we use the standard openai AsyncOpenAI client
rather than introducing a NIM-specific dep that would add nothing.

Three things this wrapper does on top of the raw client:

1. Centralizes config (base_url, model, temperature, timeout) so call sites
   don't repeat themselves and tests can swap a fake easily.

2. Tracks token usage. Every call updates a counter that LangSmith and the
   eval harness read. Cheap and useful - "what does one conversation cost?"
   is the first question anyone running an LLM in prod asks.

3. Distinguishes three call modes:
     - complete_json(): structured output, no streaming. Used by router/
       classifier nodes that need a clean JSON dict.
     - complete_stream(): token-by-token streaming. Used by the synthesizer
       node so the WebSocket can push tokens to the UI.
     - complete_with_tools(): tool-calling. Used by the free-form fallback
       where the LLM picks which Python tool to invoke.

We deliberately don't wrap LangChain's ChatNVIDIA or ChatOpenAI here.
LangChain's abstractions are convenient but add latency, change without
warning, and bury the actual request - which is exactly what the
interviewer will want to see plainly. AsyncOpenAI is the contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Any, cast
from openai import AsyncOpenAI, APIError, RateLimitError

from app.config import settings


log = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Running tally. Reset per session if you want per-session attribution."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_count: int = 0
    failed_count: int = 0
    by_model: dict[str, dict] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, model: str, usage) -> None:
        self.request_count += 1
        if usage is None:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        self.prompt_tokens += pt
        self.completion_tokens += ct

        m = self.by_model.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
        m["prompt"] += pt
        m["completion"] += ct
        m["calls"] += 1

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "failed_count": self.failed_count,
            "by_model": self.by_model,
        }


class NimClient:
    """
    Thin async wrapper around the OpenAI-compatible NIM endpoint.

    Constructed once at app boot, passed into graph nodes via state or DI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ):
        self.api_key = api_key or settings.nim_api_key
        if not self.api_key:
            raise RuntimeError(
                "NIM_API_KEY not configured. See app/config.py and .env.example"
            )

        self.base_url = base_url or settings.nim_base_url
        self.model = model or settings.nim_model
        self.timeout_s = timeout_s or settings.llm_timeout_s

        # The OpenAI client handles its own connection pooling and retries
        # (defaults: 2 retries on 5xx/429). For our purposes that's fine.
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
        )

        self.usage = UsageStats()

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Non-streaming completion that returns a parsed JSON dict.

        Used by router/classifier nodes. The system prompt is expected to
        instruct the model to emit JSON; we parse defensively in case the
        model wraps it in ``` fences or adds a preamble.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=cast(Any, [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]),
                temperature=temperature if temperature is not None else 0.0,
                max_tokens=max_tokens,
            )
        except (APIError, RateLimitError, asyncio.TimeoutError) as e:
            self.usage.failed_count += 1
            log.exception("NIM complete_json failed: %s", e)
            raise

        self.usage.record(self.model, response.usage)
        text = (response.choices[0].message.content or "").strip()

        return _parse_json_loose(text)
    async def complete_stream(
        self,
        *,
        system: str,
        user: Optional[str] = None,  # ← ADD THIS
        messages: Optional[list] = None,  # ← Make this optional
        history: Optional[list] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Streams content chunks. Yields raw token strings as they arrive.

        Either pass `user` (single user message) or `messages`/`history` (full conversation).
        Used by the synthesizer so the UI can paint tokens as they generate.
        """
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        
        # Build message list from whatever was provided
        if history:
            msg_list = history
        elif messages:
            msg_list = messages
        elif user:
            # Single user message - wrap it
            msg_list = [{"role": "user", "content": user}]
        else:
            msg_list = []
        
        for msg in msg_list:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # LangChain message object
                role = 'assistant' if msg.type == 'ai' else 'user'
                openai_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                # Already a dict, use as-is
                openai_messages.append(msg)
            else:
                # Fallback
                openai_messages.append({"role": "user", "content": str(msg)})

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}] + openai_messages,
                temperature=temperature if temperature is not None else 0.7,
                max_tokens=max_tokens,
                stream=True,
            )
        except (APIError, RateLimitError, asyncio.TimeoutError) as e:
            self.usage.failed_count += 1
            log.exception("NIM complete_stream failed: %s", e)
            raise

        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                self.usage.record(self.model, chunk.usage)
                continue
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece
    async def complete_with_tools(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict],
        history: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Tool-calling completion. Returns the raw assistant message dict so
        the caller can inspect tool_calls and dispatch them.

        Used by the free-form fallback path. The graph layer decides
        whether the LLM's tool_calls should execute or be rejected.
        """
        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                temperature=temperature if temperature is not None else 0.0,
                tools=tools,
                tool_choice="auto",
            )
        except (APIError, RateLimitError, asyncio.TimeoutError) as e:
            self.usage.failed_count += 1
            log.exception("NIM complete_with_tools failed: %s", e)
            raise

        self.usage.record(self.model, resp.usage)

        msg = resp.choices[0].message
        return {
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in (msg.tool_calls or [])
            ],
        }
def _parse_json_loose(text: str) -> dict:
    """
    Parse JSON that might be wrapped in ``` fences or have a preamble.

    LLMs sometimes ignore "return only JSON" instructions. We don't fight
    them with prompt re-engineering at runtime - we just strip the common
    garbage and parse the first object we find.
    """
    if not text or not text.strip():
        return {}

    s = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    if s.startswith("```"):
        # Drop the opening fence (possibly with language tag) and trailing fence
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()

    # Fix double-brace escaping that some LLMs do: {{...}} -> {...}
    # This is a common error when LLMs think they're in a template context
    if s.startswith("{{") and s.endswith("}}"):
        s = s[1:-1]  # strip outer layer

    # Direct parse first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Find the first {...} block and try that
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        candidate = s[start : end + 1]
        # Fix double braces in the extracted block too
        if candidate.startswith("{{") and candidate.endswith("}}"):
            candidate = candidate[1:-1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    log.warning("Could not parse JSON from LLM output: %s", s[:200])
    return {"_parse_error": True, "_raw": s[:500]}
