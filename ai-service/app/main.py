# app/main.py
import logging

from fastapi import FastAPI, HTTPException

from .llm import LLMError, MODEL, analyse
from .schemas import AnalyzeRequest, AnalyzeResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ticket Triage AI Service", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    try:
        analysis, latency_ms = await analyse(req.text, req.subject)
    except LLMError as exc:
        logger.error("analysis failed: %s", exc)
        # 502 = අපි හොඳින් ඉන්නවා, ඒත් අපේ dependency එක අවුල්
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AnalyzeResponse(analysis=analysis, model=MODEL, latency_ms=latency_ms)