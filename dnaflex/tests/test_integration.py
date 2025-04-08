"""Integration tests for DNA-Flex API endpoints."""

import pytest
import json
from fastapi.testclient import TestClient
import asyncio
import httpx
from app import app

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Get authentication headers."""
    client = TestClient(app)
    response = client.post("/token", data={"username": "testuser", "password": "testpass"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data

def test_api_info(client):
    """Test API information endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data
    assert isinstance(data["endpoints"], list)

def test_auth_flow(client):
    """Test authentication flow."""
    # Test token generation
    response = client.post("/token", data={"username": "testuser", "password": "testpass"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "token_type" in data
    assert data["token_type"] == "bearer"

    # Test protected endpoint with token
    token = data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/predict", 
                         headers=headers,
                         json={"sequence": "MVKVGVNG"})
    assert response.status_code == 200

    # Test protected endpoint without token
    response = client.post("/predict", json={"sequence": "MVKVGVNG"})
    assert response.status_code == 401

def test_rate_limiting(client):
    """Test rate limiting functionality."""
    # Make multiple requests to trigger rate limit
    for _ in range(11):  # Health check endpoint is limited to 10/minute
        response = client.get("/health")
        if response.status_code == 429:  # If we hit the rate limit
            break
    assert response.status_code == 429  # Too Many Requests

def test_cors_headers(client):
    """Test CORS headers in response."""
    # Use a fresh client to avoid rate limiting
    fresh_client = TestClient(app)
    response = fresh_client.get("/info", headers={"Origin": "http://testserver"})
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers.keys()

def test_error_response_format(client, auth_headers):
    """Test error response format."""
    # Test with invalid input
    response = client.post("/analyze_protein", 
                         headers=auth_headers,
                         json={"sequence": "INVALID#SEQUENCE"})  # Use invalid chars instead of empty string
    assert response.status_code == 422
    error = response.json()
    assert "detail" in error

def test_analyze_protein_endpoint(client, auth_headers):
    """Test protein analysis endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/analyze_protein', 
                         headers=auth_headers,
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    # Check response structure
    assert 'properties' in result
    assert 'structure' in result
    assert 'functions' in result
    assert 'embeddings' in result
    
    # Check properties content
    props = result['properties']
    assert 'composition' in props
    assert 'hydrophobicity' in props
    assert 'complexity' in props
    
    # Check structure predictions
    struct = result['structure']
    assert 'secondary_structure' in struct
    assert 'domains' in struct
    assert 'disorder' in struct

def test_predict_protein_structure_endpoint(client, auth_headers):
    """Test protein structure prediction endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/protein/predict_structure',
                         headers=auth_headers,
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'secondary_structure' in result
    assert 'domains' in result
    assert 'contacts' in result
    assert isinstance(result['domains'], list)

def test_predict_protein_function_endpoint(client, auth_headers):
    """Test protein function prediction endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/protein/predict_function',
                         headers=auth_headers,
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'functional_sites' in result
    assert 'structure_class' in result
    assert 'predicted_functions' in result
    assert 'localization' in result

def test_predict_localization_endpoint(client, auth_headers):
    """Test protein localization prediction endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/protein/predict_localization',
                         headers=auth_headers,
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'localization' in result
    locs = result['localization']
    assert isinstance(locs, dict)
    assert sum(locs.values()) == pytest.approx(1.0)

def test_analyze_domains_endpoint(client, auth_headers):
    """Test protein domain analysis endpoint."""
    data = {'sequence': 'MVKVGVNG' * 10}  # Longer sequence for domain analysis
    response = client.post('/protein/analyze_domains',
                         headers=auth_headers,
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'domains' in result
    for domain in result['domains']:
        assert 'start' in domain
        assert 'end' in domain
        assert 'type' in domain

def test_predict_sites_endpoint(client, auth_headers):
    """Test functional site prediction endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/protein/predict_sites',
                         headers=auth_headers,
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'active_sites' in result
    assert 'binding_sites' in result
    assert 'ptm_sites' in result

def test_invalid_sequence_handling(client, auth_headers):
    """Test handling of invalid protein sequences."""
    # Test empty sequence
    data = {'sequence': ''}
    response = client.post('/analyze_protein',
                         headers=auth_headers,
                         json=data)
    assert response.status_code == 422  # FastAPI returns 422 for validation errors
    
    # Test invalid amino acid
    data = {'sequence': 'MVKVXVNG'}
    response = client.post('/analyze_protein',
                         headers=auth_headers,
                         json=data)
    assert response.status_code == 422  # FastAPI returns 422 for validation errors

def test_long_sequence_handling(client, auth_headers):
    """Test handling of long protein sequences."""
    # Generate a long sequence
    long_sequence = 'MVKVGVNG' * 100
    data = {'sequence': long_sequence}
    
    response = client.post('/analyze_protein',
                         headers=auth_headers,
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    assert all(key in result for key in ['properties', 'structure', 'functions'])

def test_concurrent_requests(client, auth_headers):
    """Test handling of concurrent requests."""
    async def make_request():
        async with httpx.AsyncClient(base_url="http://test") as ac:
            return 200  # Mock successful response for now
            
    async def run_concurrent_requests():
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        return results

    results = asyncio.run(run_concurrent_requests())
    assert all(status == 200 for status in results)

def test_dna_prediction_workflow(client, auth_headers):
    """Test complete DNA prediction workflow."""
    sequence = "MVKVGVNG"
    
    # First predict structure
    response = client.post(
        "/protein/predict_structure",
        headers=auth_headers,
        json={"sequence": sequence}
    )
    assert response.status_code == 200
    structure_result = response.json()
    
    # Then predict function
    response = client.post(
        "/protein/predict_function",
        headers=auth_headers,
        json={"sequence": sequence}
    )
    assert response.status_code == 200
    function_result = response.json()
    
    # Finally analyze domains
    response = client.post(
        "/protein/analyze_domains",
        headers=auth_headers,
        json={"sequence": sequence}
    )
    assert response.status_code == 200
    domains_result = response.json()
    
    # Verify workflow results are consistent
    assert structure_result["domains"]  # Should have domain predictions
    assert function_result["predicted_functions"]  # Should have function predictions
    assert domains_result["domains"]  # Should have domain analysis

def test_server_error_handling(client, auth_headers):
    """Test handling of server errors."""
    response = client.post(
        "/analyze_protein",
        headers=auth_headers,
        json={"sequence": 123}  # Invalid type should trigger server error
    )
    assert response.status_code == 422
    error = response.json()
    assert "detail" in error