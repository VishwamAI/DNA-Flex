"""Direct tests for DNA sequence analysis without server dependencies."""

from dnaflex.models.analysis import analyze

def test_dna_analysis():
    """Test direct DNA sequence analysis."""
    # Test sequence with known properties
    sequence = "ATGCATGCATGC"
    
    result = analyze(sequence)
    
    # Basic properties
    assert result['length'] == 12
    assert result['gc_content'] == 50.0
    
    # Composition
    assert result['base_composition']['A'] == 3
    assert result['base_composition']['T'] == 3
    assert result['base_composition']['G'] == 3
    assert result['base_composition']['C'] == 3
    
    # Structural properties
    assert len(result['stability_scores']) == len(sequence)
    assert all(0 <= score <= 1 for score in result['stability_scores'])
    
    # Sequence complexity
    assert 'shannon_entropy' in result['sequence_complexity']
    assert 0 <= result['sequence_complexity']['shannon_entropy'] <= 2.0
    
    # Quality metrics
    assert 'base_balance' in result['quality_metrics']
    assert 'sequence_quality' in result['quality_metrics']
    assert all(0 <= score <= 1 for score in result['quality_metrics'].values())

def test_repeat_detection():
    """Test repeat sequence detection."""
    # Sequence with known repeats
    sequence = "ATGCATGC"  # Direct repeat
    
    result = analyze(sequence)
    repeats = result['repeats']
    
    # Should find the ATGC repeat
    direct_repeats = repeats['direct']
    assert any(repeat['pattern'] == "ATGC" for repeat in direct_repeats)

def test_invalid_sequence():
    """Test handling of invalid sequences."""
    try:
        analyze("ATGX")  # Invalid base X
        assert False, "Should raise ValueError for invalid sequence"
    except ValueError:
        pass  # Expected behavior

def test_motif_detection():
    """Test motif detection."""
    # Sequence with TATA box
    sequence = "GCTATAAAT"
    
    result = analyze(sequence)
    motifs = result['motifs']
    
    assert 'regulatory' in motifs
    assert any(motif['type'] == 'TATA_box' for motif in motifs['regulatory'])