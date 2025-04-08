"""Tests for BioLLM protein sequence analysis functionality."""

import pytest
import numpy as np
from dnaflex.models.dna_llm import BioLLM
import jax as jnp
@pytest.fixture
def protein_model():
    """Create protein model instance for testing."""
    return BioLLM(model_type='protein')

def test_model_initialization(protein_model):
    """Test protein model initialization."""
    assert protein_model.model_type == 'protein'
    assert protein_model.vocab == list("ACDEFGHIKLMNPQRSTVWY")
    assert len(protein_model.token_to_id) == len(protein_model.vocab) + len(protein_model.special_tokens)

def test_tokenization(protein_model):
    """Test protein sequence tokenization."""
    sequence = "MVKVGVNG"
    tokens = protein_model.tokenize(sequence)
    
    # Check token array properties
    assert tokens.shape == (1, protein_model.max_seq_length)  # Shape should be (1, max_seq_length)
    assert tokens.ndim == 2  # Should be a 2D array

def test_embedding_generation(protein_model):
    """Test protein sequence embedding generation."""
    sequence = "MVKVGVNG"
    embeddings = protein_model.generate_embeddings(sequence)
    
    # Check embedding dimensions
    assert embeddings.shape[0] == len(sequence) + 2  # +2 for CLS and SEP
    assert embeddings.shape[1] == protein_model.embedding_size

def test_protein_property_analysis(protein_model):
    """Test protein property analysis."""
    sequence = "MVKVGVNG"
    properties = protein_model._analyze_protein_properties(sequence)
    
    # Check composition
    assert 'composition' in properties
    assert sum(properties['composition'].values()) == pytest.approx(1.0)
    
    # Check hydrophobicity
    assert 'hydrophobicity' in properties
    assert -5.0 <= properties['hydrophobicity'] <= 5.0
    
    # Check complexity
    assert 'complexity' in properties
    assert 0.0 <= properties['complexity'] <= 1.0
    
    # Check secondary structure propensities
    assert 'secondary_structure_propensities' in properties
    assert len(properties['secondary_structure_propensities']['helix']) == len(sequence)
    assert len(properties['secondary_structure_propensities']['sheet']) == len(sequence)

def test_structure_prediction(protein_model):
    """Test protein structure prediction."""
    sequence = "MVKVGVNG"
    structure = protein_model._predict_protein_structure(sequence)
    
    # Check secondary structure prediction
    assert 'secondary_structure' in structure
    assert all(0.0 <= p <= 1.0 for p in structure['secondary_structure']['helix'])
    assert all(0.0 <= p <= 1.0 for p in structure['secondary_structure']['sheet'])
    
    # Check domain prediction
    assert 'domains' in structure
    for domain in structure['domains']:
        assert 'start' in domain
        assert 'end' in domain
        assert 'type' in domain
        assert domain['start'] < domain['end']
    
    # Check disorder prediction
    assert 'disorder' in structure
    assert all(0.0 <= d <= 1.0 for d in structure['disorder'])
    
    # Check contact prediction
    assert 'contacts' in structure
    for contact in structure['contacts']:
        assert len(contact) == 3  # (i, j, probability)
        assert contact[0] < contact[1]  # i < j
        assert 0.0 <= contact[2] <= 1.0  # probability

def test_function_prediction(protein_model):
    """Test protein function prediction."""
    sequence = "MVKVGVNG"
    functions = protein_model._predict_protein_functions(sequence)
    
    # Check functional sites
    assert 'functional_sites' in functions
    assert 'active_sites' in functions['functional_sites']
    assert 'binding_sites' in functions['functional_sites']
    assert 'ptm_sites' in functions['functional_sites']
    
    # Check structure class prediction
    assert 'structure_class' in functions
    assert sum(functions['structure_class'].values()) == pytest.approx(1.0)
    
    # Check function class prediction
    assert 'predicted_functions' in functions
    assert sum(functions['predicted_functions'].values()) == pytest.approx(1.0)
    
    # Check localization prediction
    assert 'localization' in functions
    assert sum(functions['localization'].values()) == pytest.approx(1.0)

def test_invalid_sequences(protein_model):
    """Test handling of invalid protein sequences."""
    # Test with invalid amino acid
    with pytest.raises(ValueError):
        protein_model.analyze_protein("MVKVBVNG")
    
    # Test with empty sequence
    with pytest.raises(ValueError):
        protein_model.analyze_protein("")

def test_pattern_matching(protein_model):
    """Test protein sequence pattern matching."""
    # Test exact match
    assert protein_model._match_pattern("SER", "SER")
    
    # Test wildcard
    assert protein_model._match_pattern("ABC", "AXC")
    
    # Test character class
    assert protein_model._match_pattern("ABC", "A[BC]C")
    assert not protein_model._match_pattern("ADC", "A[BC]C")

def test_localization_features(protein_model):
    """Test protein localization feature detection."""
    # Test signal peptide
    signal_seq = "MKLLVLLFVLLFLVQVSSA" + "A" * 20  # Classic signal peptide
    assert protein_model._has_signal_peptide(signal_seq)
    
    # Test nuclear localization
    nls_seq = "MVKVPKKKRKVA"  # Classic NLS
    assert protein_model._has_nuclear_features(nls_seq)
    
    # Test transmembrane
    tm_seq = "LLLLLLLLLLLLLLLLLLLL" + "A" * 20  # Hydrophobic stretch
    assert protein_model._has_transmembrane_features(tm_seq)

def test_long_sequence_handling(protein_model):
    """Test handling of long protein sequences."""
    # Generate long sequence
    long_sequence = "MVKVGVNG" * 100
    
    # Test tokenization
    tokens = protein_model.tokenize(long_sequence)
    assert tokens.shape == (1, protein_model.max_seq_length)  # Shape should be (1, max_seq_length)
    
    # Test embedding generation
    embeddings = protein_model.generate_embeddings(long_sequence)
    assert embeddings.shape[0] <= protein_model.max_seq_length

def test_next_token_prediction(protein_model):
    """Test next token prediction."""
    sequence = "MVKVGVN"
    predictions = protein_model.predict_next_tokens(sequence, num_predictions=3)
    
    assert len(predictions) == 3
    assert all(p in protein_model.vocab + protein_model.special_tokens for p in predictions)

def test_sequence_complexity(protein_model):
    """Test sequence complexity calculation."""
    # Simple repetitive sequence
    simple_seq = "AAAA"
    simple_complexity = protein_model._calculate_protein_complexity(simple_seq)
    
    # Complex sequence
    complex_seq = "ARNDCEQGH"
    complex_complexity = protein_model._calculate_protein_complexity(complex_seq)
    
    assert simple_complexity < complex_complexity