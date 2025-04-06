"""Tests for DNA-Flex data management."""

import pytest
from pathlib import Path
import tempfile
import shutil
import json
import numpy as np
from dnaflex.data.manager import DataManager
from dnaflex.data.cache import DataCache
from dnaflex.data.loader import DataLoader

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def data_manager(temp_dir):
    """Create DataManager instance with temporary directories."""
    return DataManager(
        data_dir=temp_dir / 'data',
        cache_dir=temp_dir / 'cache',
        email='test@example.com'
    )

def test_data_manager_initialization(data_manager, temp_dir):
    """Test DataManager initialization."""
    assert data_manager.data_dir.exists()
    assert data_manager.sequences_dir.exists()
    assert data_manager.structures_dir.exists()
    assert data_manager.results_dir.exists()

def test_save_and_load_sequence(data_manager):
    """Test saving and loading sequence data."""
    sequence_data = {
        'sequence': 'ATCGATCG',
        'metadata': {'id': 'test_seq'}
    }
    
    # Test FASTA format
    fasta_path = data_manager.save_sequence(sequence_data, 'test_seq', 'fasta')
    assert fasta_path.exists()
    
    # Test JSON format
    json_path = data_manager.save_sequence(sequence_data, 'test_seq', 'json')
    assert json_path.exists()
    
    # Test loading
    loaded_data = data_manager.get_sequence('test_seq')
    assert 'ATCG' in ''.join(loaded_data.values())

def test_save_and_load_structure(data_manager):
    """Test saving and loading structure data."""
    structure_data = {
        'chains': {
            'A': [{'id': 1, 'name': 'DA', 'atoms': {'P': {'coords': [1.0, 2.0, 3.0]}}}]
        }
    }
    
    path = data_manager.save_structure(structure_data, 'test_struct')
    assert path.exists()
    
    # Load and verify
    loaded = data_manager.loader.load_json(path)
    assert loaded['chains']['A'][0]['atoms']['P']['coords'] == [1.0, 2.0, 3.0]

def test_cache_operations(data_manager):
    """Test cache operations."""
    test_data = {'test': 'data'}
    
    # Set and get
    data_manager.cache.set('test_key', test_data)
    cached = data_manager.cache.get('test_key')
    assert cached == test_data
    
    # Invalidate
    data_manager.cache.invalidate('test_key')
    assert data_manager.cache.get('test_key') is None

def test_results_management(data_manager):
    """Test saving and retrieving analysis results."""
    test_results = {
        'analysis': 'test',
        'data': [1, 2, 3]
    }
    
    # Save results
    path = data_manager.save_results(
        test_results,
        analysis_type='test_analysis',
        identifier='test_id'
    )
    assert path.exists()
    
    # Retrieve results
    results = data_manager.get_results(identifier='test_id')
    assert len(results) == 1
    assert results[0]['data']['analysis'] == 'test'

def test_data_loader_operations(temp_dir):
    """Test DataLoader operations."""
    loader = DataLoader(temp_dir)
    
    # Test NumPy data
    test_arrays = {
        'array1': np.array([1, 2, 3]),
        'array2': np.array([[1, 2], [3, 4]])
    }
    
    numpy_path = temp_dir / 'test.npz'
    loader.save_numpy(test_arrays, numpy_path)
    loaded_arrays = loader.load_numpy(numpy_path)
    
    assert np.array_equal(loaded_arrays['array1'], test_arrays['array1'])
    assert np.array_equal(loaded_arrays['array2'], test_arrays['array2'])
    
    # Test JSON data
    test_data = {'test': 'data', 'numbers': [1, 2, 3]}
    json_path = temp_dir / 'test.json'
    loader.save_json(test_data, json_path)
    loaded_data = loader.load_json(json_path)
    
    assert loaded_data == test_data

def test_data_export_import(data_manager, temp_dir):
    """Test data export and import functionality."""
    # Create some test data
    sequence_data = {'sequence': 'ATCG'}
    structure_data = {'structure': 'test'}
    results_data = {'results': 'test'}
    
    data_manager.save_sequence(sequence_data, 'test_seq')
    data_manager.save_structure(structure_data, 'test_struct')
    data_manager.save_results(results_data, 'test', 'test_id')
    
    # Export data
    export_dir = temp_dir / 'export'
    data_manager.export_data(export_dir, include_cache=True)
    
    # Create new data manager and import
    new_data_dir = temp_dir / 'new_data'
    new_cache_dir = temp_dir / 'new_cache'
    new_manager = DataManager(new_data_dir, new_cache_dir)
    new_manager.import_data(export_dir, include_cache=True)
    
    # Verify imported data
    assert list(new_manager.sequences_dir.glob('*'))
    assert list(new_manager.structures_dir.glob('*'))
    assert list(new_manager.results_dir.glob('*'))