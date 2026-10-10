import json
import os
import re
import time
import asyncio
import logging
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


async def _call_once(client: httpx.AsyncClient, prompt: str) -> str:
    """එක HTTP call එකක්. Raw text එක return කරනවා."""
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

    resp = await client.post(API_URL, json=request_body, timeout=TIMEOUT_S)
    resp.raise_for_status()

    data = resp.json()
    return data["message"]["content"]


async def analyse(text: str, subject: str | None = None) -> Tuple[TicketAnalysis, int]:
    """
    Ticket එකක් analyse කරනවා.

    තාවකාලික අවුල් (bad JSON, schema fail, 5xx, 429, network)
    retry කරනවා. ස්ථිර අවුල් (400, 401, 404) retry කරන්නේ නෑ.
    """
    prompt = f"Subject: {subject}\n\nTicket:\n{text}" if subject else f"Ticket:\n{text}"
    started = time.perf_counter()
    last_error: Exception | None = None

    async with httpx.AsyncClient() as client:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw = await _call_once(client, prompt)
                payload = _extract_json(raw)
                analysis = TicketAnalysis.model_validate(payload)

                latency_ms = int((time.perf_counter() - started) * 1000)
                if attempt > 1:
                    logger.info("succeeded on attempt %d", attempt)
                return analysis, latency_ms

            except (ValidationError, ValueError, json.JSONDecodeError, KeyError) as exc:
                # Model එක වැරදි shape එකක් දුන්නා. ආයෙ sample කළාම
                # බොහෝ විට හරියනවා — retry කරන්න වටිනවා.
                last_error = exc
                logger.warning("attempt %d: bad output (%s)", attempt, exc)

            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                if status < 500 and status != 429:
                    # 4xx (429 හැර) retry එකෙන් හරියන්නේ නෑ.
                    raise LLMError(f"LLM rejected the request: {status}") from exc
                logger.warning("attempt %d: HTTP %d", attempt, status)

            except httpx.RequestError as exc:
                # Network / timeout — තාවකාලික වෙන්න පුළුවන්.
                last_error = exc
                logger.warning("attempt %d: network error (%s)", attempt, exc)

            if attempt < MAX_ATTEMPTS:
                delay = 0.5 * (2 ** (attempt - 1))   # 0.5s, 1s, 2s ...
                logger.info("retrying in %.1fs", delay)
                await asyncio.sleep(delay)

    raise LLMError(f"failed after {MAX_ATTEMPTS} attempts: {last_error}")

    