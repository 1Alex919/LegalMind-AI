"""Pydantic request/response models for the API."""

from pydantic import BaseModel


# --- Upload ---

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    chunks_stored: int
    message: str


# --- Risk Analysis ---

class RiskRequest(BaseModel):
    document_id: str


class RiskItem(BaseModel):
    risk_type: str
    severity: str
    clause_text: str
    explanation: str
    recommendation: str
    page: int | None = None


class RiskResponse(BaseModel):
    risks: list[RiskItem]
    summary: str
    total_risks: int
    document_id: str


# --- Summary ---

class SummaryRequest(BaseModel):
    document_id: str


class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str]
    parties: list[str]
    contract_type: str
    document_id: str


# --- Q&A ---

class QueryRequest(BaseModel):
    document_id: str
    question: str


class SourceItem(BaseModel):
    text: str
    page: int | None = None
    relevance_score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[SourceItem]
    reasoning: str
    document_id: str


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    version: str
