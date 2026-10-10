# app/schemas.py
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    COMPLAINT = "complaint"
    OTHER = "other"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class AnalyzeRequest(BaseModel):
    """Client එකෙන් එන දේ."""
    text: str = Field(min_length=10, max_length=5000)
    subject: Optional[str] = Field(default=None, max_length=200)


class TicketAnalysis(BaseModel):
    """LLM එකෙන් එන්න ඕන දේ — හරියටම මේ shape එකට."""
    category: Category
    priority: Priority
    summary: str = Field(min_length=5, max_length=300)
    entities: List[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=5, max_length=500)


class AnalyzeResponse(BaseModel):
    analysis: TicketAnalysis
    model: str
    cached: bool = False
    latency_ms: int