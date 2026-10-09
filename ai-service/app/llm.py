import json
import os
import re
import time
from typing import Tuple

import httpx
from pydantic import ValidationError

from .schemas import TicketAnalysis

MODEL = os.getenv("LLM_MODEL", "llama3.2")
API_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
TIMEOUT_S = 120.0   # local model එක cloud එකට වඩා සෙමින්


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
    """```json fences හෝ අමතර වාක්‍ය ඉවත් කරලා JSON එක ගන්නවා."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("no JSON object found in response")


async def analyse(text: str, subject: str | None = None) -> Tuple[TicketAnalysis, int]:
    """Ticket එකක් analyse කරනවා. (analysis, latency_ms) return කරනවා."""
    prompt = f"Subject: {subject}\n\nTicket:\n{text}" if subject else f"Ticket:\n{text}"
    started = time.perf_counter()

    request_body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(API_URL, json=request_body, timeout=TIMEOUT_S)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM returned {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except httpx.RequestError as exc:
            raise LLMError(f"could not reach Ollama at {API_URL}: {exc}") from exc

    data = resp.json()
    try:
        raw = data["message"]["content"]
    except KeyError as exc:
        raise LLMError(f"unexpected response shape: {exc}") from exc

    try:
        payload = _extract_json(raw)
        analysis = TicketAnalysis.model_validate(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise LLMError(f"could not parse LLM output: {exc}") from exc
    except ValidationError as exc:
        raise LLMError(f"LLM output did not match schema: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    return analysis, latency_ms