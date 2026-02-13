# API Documentation

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

## Endpoints

### POST /upload

Upload a PDF or DOCX document for analysis.

**Request**: `multipart/form-data` with `file` field.

**Response**:
```json
{
  "document_id": "uuid",
  "filename": "contract.pdf",
  "total_pages": 5,
  "chunks_stored": 23,
  "message": "Document processed: 23 chunks indexed"
}
```

### POST /analyze/risks

Run risk analysis on an uploaded document.

**Request**:
```json
{
  "document_id": "uuid"
}
```

**Response**:
```json
{
  "risks": [
    {
      "risk_type": "liability",
      "severity": "high",
      "clause_text": "The contractor shall be liable for all damages...",
      "explanation": "Unlimited liability clause exposes significant financial risk.",
      "recommendation": "Negotiate a liability cap.",
      "page": 3
    }
  ],
  "summary": "The contract has 3 high-severity risks...",
  "total_risks": 5,
  "document_id": "uuid"
}
```

### POST /analyze/summary

Generate a summary of an uploaded document.

**Request**:
```json
{
  "document_id": "uuid"
}
```

**Response**:
```json
{
  "summary": "This is a Non-Disclosure Agreement between...",
  "key_points": ["2-year term", "Mutual obligations", "30-day termination notice"],
  "parties": ["Company A", "Company B"],
  "contract_type": "NDA",
  "document_id": "uuid"
}
```

### POST /query

Ask a question about an uploaded document.

**Request**:
```json
{
  "document_id": "uuid",
  "question": "What is the termination period?"
}
```

**Response**:
```json
{
  "answer": "The termination period is 30 days written notice...",
  "confidence": 0.92,
  "sources": [
    {
      "text": "Either party may terminate with 30 days...",
      "page": 5,
      "relevance_score": 0.89
    }
  ],
  "reasoning": "Used QA Agent with 3 sources",
  "document_id": "uuid"
}
```

### GET /health

Health check.

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

## Error Responses

All errors return:
```json
{
  "detail": "Error message description"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (unsupported file type, empty question) |
| 404 | Document not found |
| 500 | Internal processing error |
