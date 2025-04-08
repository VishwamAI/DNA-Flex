# DNA-Flex API Documentation

## Overview

The DNA-Flex API provides a comprehensive interface for DNA sequence analysis, structure prediction, and flexibility assessment. This RESTful API is built with FastAPI and supports async operations, authentication, and rate limiting.

## Authentication

All protected endpoints require JWT authentication. To authenticate:

1. Obtain a token using the `/token` endpoint
2. Include the token in the Authorization header: `Bearer <token>`

```bash
curl -X POST "http://localhost:8000/token" \
     -d "username=your-username&password=your-password"
```

## Rate Limiting

- Standard endpoints: 5 requests/minute
- Health check: 10 requests/minute
- Batch processing: 2 requests/minute

## Endpoints

### Authentication

#### POST /token

Get authentication token.

**Request:**
```json
{
    "username": "string",
    "password": "string"
}
```

**Response:**
```json
{
    "access_token": "string",
    "token_type": "bearer"
}
```

### System Information

#### GET /health

Check API health status.

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2025-04-08T13:41:18.371Z"
}
```

#### GET /info

Get API information and available endpoints.

**Response:**
```json
{
    "name": "DNA-Flex API",
    "version": "1.0.0",
    "description": "Advanced DNA sequence analysis and flexibility prediction",
    "endpoints": [
        {
            "path": "/predict",
            "method": "POST",
            "description": "Predict DNA sequence properties"
        },
        ...
    ]
}
```

### Analysis Endpoints

#### POST /predict

Predict DNA sequence properties with async background processing.

**Request:**
```json
{
    "sequence": "ATGCATGCATGC",
    "description": "Sample sequence"
}
```

**Response:**
```json
{
    "request_id": "uuid-string",
    "status": "pending",
    "created_at": "2025-04-08T13:41:18.371Z"
}
```

#### GET /tasks/{task_id}

Get the status and results of an analysis task.

**Response:**
```json
{
    "request_id": "uuid-string",
    "status": "completed",
    "result": {
        "analysis": {
            "gc_content": 50.0,
            "stability_scores": [...],
            "motifs": [...],
            "secondary_structures": [...]
        },
        "dynamics": {
            "rmsd": [...],
            "rmsf": [...],
            "energies": {...}
        },
        "variations": [...],
        "binding_sites": [...],
        "mutations": [...],
        "nlp_insights": {...}
    },
    "created_at": "2025-04-08T13:41:18.371Z",
    "completed_at": "2025-04-08T13:41:20.371Z"
}
```

#### POST /analyze

Direct DNA sequence analysis (synchronous).

**Request:**
```json
{
    "sequence": "ATGCATGCATGC",
    "description": "Sample sequence"
}
```

**Response:**
```json
{
    "sequence": "ATGCATGCATGC",
    "analysis": {
        "length": 12,
        "gc_content": 50.0,
        "base_composition": {
            "A": 0.25,
            "T": 0.25,
            "G": 0.25,
            "C": 0.25
        },
        "stability_scores": [...],
        "sequence_complexity": {...},
        "motifs": [...],
        "secondary_structures": [...],
        "repeats": {...}
    },
    "timestamp": "2025-04-08T13:41:18.371Z"
}
```

#### POST /batch

Batch analyze multiple DNA sequences.

**Request:**
- Multipart form data with file upload
- Optional query parameter: `max_sequences` (default: 10, max: 100)

**Response:**
```json
{
    "total_sequences": 5,
    "processed": 5,
    "results": [
        {
            "sequence": "ATGC...",
            "analysis": {...}
        },
        ...
    ],
    "timestamp": "2025-04-08T13:41:18.371Z"
}
```

### Statistics

#### GET /stats

Get API usage statistics.

**Response:**
```json
{
    "total_requests": 100,
    "completed_tasks": 95,
    "failed_tasks": 5,
    "timestamp": "2025-04-08T13:41:18.371Z"
}
```

## Error Responses

All endpoints return standard HTTP status codes and a consistent error response format:

```json
{
    "detail": "Error message",
    "status_code": 400,
    "timestamp": "2025-04-08T13:41:18.371Z",
    "path": "/endpoint",
    "method": "POST"
}
```

Common status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

## Data Models

### SequenceInput
```json
{
    "sequence": "string",
    "description": "string (optional)"
}
```

### AnalysisResult
```json
{
    "request_id": "string",
    "status": "string",
    "result": "object (optional)",
    "error": "string (optional)",
    "created_at": "datetime",
    "completed_at": "datetime (optional)"
}
```

## Examples

### Python Client Example

```python
import requests

class DNAFlexClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.token = None
    
    def authenticate(self, username, password):
        response = requests.post(
            f"{self.base_url}/token",
            data={"username": username, "password": password}
        )
        self.token = response.json()["access_token"]
        
    def predict(self, sequence, description=None):
        headers = {"Authorization": f"Bearer {self.token}"}
        data = {"sequence": sequence}
        if description:
            data["description"] = description
            
        response = requests.post(
            f"{self.base_url}/predict",
            headers=headers,
            json=data
        )
        return response.json()
    
    def get_task_status(self, task_id):
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(
            f"{self.base_url}/tasks/{task_id}",
            headers=headers
        )
        return response.json()

# Usage
client = DNAFlexClient()
client.authenticate("username", "password")
result = client.predict("ATGCATGCATGC")
task_id = result["request_id"]

# Poll for results
while True:
    status = client.get_task_status(task_id)
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(1)
```

## WebSocket Support (Coming Soon)

Real-time task status updates will be available through WebSocket connections at `/ws/tasks/{task_id}`.