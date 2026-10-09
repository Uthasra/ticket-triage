# app/llm.py
import json
import os
import re
import time
from typing import Tuple

import httpx
from pydantic import ValidationError

from .schemas import TicketAnalysis

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TIMEOUT_S = 30.0


SYSTEM_PROMPT = """You are a support ticket triage system.

Analyse the ticket and respond with ONLY a JSON object. No markdown fences,
no explanation before or after. The object must have exactly these keys:

{
  "category": one of ["billing","technical","account","feature_request","complaint","other"],
  "priority": one of ["low","medium","high","urgent"],
  "summary": a one-sentence summary, 5 to 300 characters,
  "entities": array of up to 10 short strings - product names, error codes, order ids,
  "confidence": a number between 0.0 and 1.0,
  "reasoning": one or two sentences explaining the category and priority, max 500 characters
}

Priority guidance:
- urgent: service is down, data loss, security issue, or payment taken in error
- high: a core feature is broken for this user and there is no workaround
- medium: something is wrong but the user can still work
- low: questions, cosmetic issues, feature requests
"""


class LLMError(Exception):
    """LLM එකෙන් හරි analysis එකක් ගන්න බැරි වුනාම."""


def _extract_json(text: str) -> dict:
    """
    LLM එක සමහර වෙලාවට ```json fence එකක් ඇතුළේ JSON එක දානවා,
    නැත්තං ඉස්සරහින් වාක්‍යයක් දානවා. ඒවා ඉවත් කරනවා.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: පළවෙනි {...} block එක හොයනවා
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("no JSON object found in response")


async def analyse(text: str, subject: str | None = None) -> Tuple[TicketAnalysis, int]:
    """Ticket එකක් analyse කරනවා. (analysis, latency_ms) return කරනවා."""
    if not API_KEY:
        raise LLMError("ANTHROPIC_API_KEY is not set")

    prompt = f"Subject: {subject}\n\nTicket:\n{text}" if subject else f"Ticket:\n{text}"
    started = time.perf_counter()

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                API_URL,
                headers={
                    "content-type": "application/json",
                    "x-api-key": API_KEY,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 1000,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=TIMEOUT_S,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM returned {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise LLMError(f"could not reach LLM: {exc}") from exc

    # Response එකේ text blocks එකතු කරනවා
    data = resp.json()
    raw = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )

    # Parse + validate
    try:
        payload = _extract_json(raw)
        analysis = TicketAnalysis.model_validate(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise LLMError(f"could not parse LLM output: {exc}") from exc
    except ValidationError as exc:
        raise LLMError(f"LLM output did not match schema: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    return analysis, latency_ms