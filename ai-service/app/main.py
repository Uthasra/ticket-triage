import hashlib
import logging
import os
from collections import OrderedDict

from fastapi import FastAPI, HTTPException

from .llm import LLMError, MODEL, analyse
from .schemas import AnalyzeRequest, AnalyzeResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ticket Triage AI Service", version="0.2.0")

CACHE_MAX = int(os.getenv("CACHE_MAX", "500"))
_cache: "OrderedDict[str, dict]" = OrderedDict()


def _cache_key(text: str, subject: str | None) -> str:
    """Text එක hash කරනවා — key එක කෙටි, memory එක ඉතිරි."""
    return hashlib.sha256(f"{subject or ''}||{text}".encode()).hexdigest()


def _cache_get(key: str):
    if key in _cache:
        _cache.move_to_end(key)      # LRU: දැන් පාවිච්චි කළා කියලා මාක් කරනවා
        return _cache[key]
    return None


def _cache_put(key: str, value: dict) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)   # වැඩිම කාලයක් පාවිච්චි නොකළ එක අයින්


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "cache_size": len(_cache)}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    key = _cache_key(req.text, req.subject)

    hit = _cache_get(key)
    if hit is not None:
        logger.info("cache hit")
        return AnalyzeResponse(
            analysis=hit["analysis"], model=MODEL, cached=True, latency_ms=0
        )

    try:
        analysis, latency_ms = await analyse(req.text, req.subject)
    except LLMError as exc:
        logger.error("analysis failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _cache_put(key, {"analysis": analysis})
    return AnalyzeResponse(
        analysis=analysis, model=MODEL, cached=False, latency_ms=latency_ms
    )