"""Integration tests for DNA-Flex API endpoints."""

import pytest
import json
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

def test_analyze_protein_endpoint(client):
    """Test protein analysis endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/analyze_protein', 
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

def test_predict_protein_structure_endpoint(client):
    """Test protein structure prediction endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/protein/predict_structure',
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'secondary_structure' in result
    assert 'domains' in result
    assert 'contacts' in result
    assert isinstance(result['domains'], list)

def test_predict_protein_function_endpoint(client):
    """Test protein function prediction endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/protein/predict_function',
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'functional_sites' in result
    assert 'structure_class' in result
    assert 'predicted_functions' in result
    assert 'localization' in result

def test_predict_localization_endpoint(client):
    """Test protein localization prediction endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/protein/predict_localization',
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'localization' in result
    locs = result['localization']
    assert isinstance(locs, dict)
    assert sum(locs.values()) == pytest.approx(1.0)

def test_analyze_domains_endpoint(client):
    """Test protein domain analysis endpoint."""
    data = {'sequence': 'MVKVGVNG' * 10}  # Longer sequence for domain analysis
    response = client.post('/protein/analyze_domains',
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'domains' in result
    for domain in result['domains']:
        assert 'start' in domain
        assert 'end' in domain
        assert 'type' in domain

def test_predict_sites_endpoint(client):
    """Test functional site prediction endpoint."""
    data = {'sequence': 'MVKVGVNG'}
    response = client.post('/protein/predict_sites',
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    
    assert 'active_sites' in result
    assert 'binding_sites' in result
    assert 'ptm_sites' in result

def test_invalid_sequence_handling(client):
    """Test handling of invalid protein sequences."""
    # Test empty sequence
    data = {'sequence': ''}
    response = client.post('/analyze_protein',
                         json=data)
    assert response.status_code == 422  # FastAPI returns 422 for validation errors
    
    # Test invalid amino acid
    data = {'sequence': 'MVKVXVNG'}
    response = client.post('/analyze_protein',
                         json=data)
    assert response.status_code == 422  # FastAPI returns 422 for validation errors

def test_long_sequence_handling(client):
    """Test handling of long protein sequences."""
    # Generate a long sequence
    long_sequence = 'MVKVGVNG' * 100
    data = {'sequence': long_sequence}
    
    response = client.post('/analyze_protein',
                         json=data)
    
    assert response.status_code == 200
    result = response.json()
    assert all(key in result for key in ['properties', 'structure', 'functions'])