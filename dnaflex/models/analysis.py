"""DNA sequence analysis module."""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import re

def analyze(sequence: str) -> Dict[str, Any]:
    """Analyze DNA sequence characteristics.
    
    Performs comprehensive DNA sequence analysis including:
    - Base composition and GC content
    - Sequence complexity
    - Local stability
    - Motif detection
    - Secondary structure prediction
    - Repeats analysis
    """
    # Basic sequence validation
    if not sequence or not _is_valid_dna(sequence):
        raise ValueError("Invalid DNA sequence")
        
    # Calculate base composition
    base_counts = {
        'A': sequence.count('A'),
        'T': sequence.count('T'),
        'G': sequence.count('G'),
        'C': sequence.count('C')
    }
    
    gc_content = (base_counts['G'] + base_counts['C']) / len(sequence) * 100
    
    # Calculate local structural properties
    stability_scores = _calculate_stability_scores(sequence)
    
    # Calculate sequence complexity
    complexity = _calculate_sequence_complexity(sequence)
    
    # Find sequence motifs
    motifs = _find_sequence_motifs(sequence)
    
    # Predict secondary structures
    secondary_structures = _predict_secondary_structures(sequence)
    
    # Analyze repeats
    repeats = _analyze_repeats(sequence)
    
    return {
        'length': len(sequence),
        'base_composition': base_counts,
        'gc_content': gc_content,
        'stability_scores': stability_scores.tolist(),
        'sequence_complexity': complexity,
        'motifs': motifs,
        'secondary_structures': secondary_structures,
        'repeats': repeats,
        'quality_metrics': _calculate_quality_metrics(sequence)
    }

def _is_valid_dna(sequence: str) -> bool:
    """Validate DNA sequence."""
    return bool(sequence) and all(base in 'ACGT' for base in sequence.upper())

def _calculate_stability_scores(sequence: str) -> np.ndarray:
    """Calculate local structural stability scores."""
    # Nearest-neighbor stability model with improved parameters
    stability_map = {
        'AA': 1.0, 'AT': 0.8, 'AG': 0.9, 'AC': 0.8,
        'TA': 0.6, 'TT': 1.0, 'TG': 0.7, 'TC': 0.7,
        'GA': 0.9, 'GT': 0.7, 'GG': 1.2, 'GC': 1.1,
        'CA': 0.8, 'CT': 0.7, 'CG': 1.1, 'CC': 1.2
    }
    
    scores = np.zeros(len(sequence))
    
    # Calculate sliding window stability
    window_size = 4
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        window_score = 0
        for j in range(len(window)-1):
            dinuc = window[j:j+2]
            window_score += stability_map.get(dinuc, 0.5)
        scores[i:i+window_size] += window_score / window_size
        
    # Normalize scores
    if len(scores) > 0:
        scores = scores / np.max(scores)
    
    return scores

def _calculate_sequence_complexity(sequence: str) -> Dict[str, float]:
    """Calculate various sequence complexity metrics."""
    # Linguistic complexity
    k_values = range(1, 6)
    observed_kmers = {k: set() for k in k_values}
    possible_kmers = {k: min(4**k, len(sequence)-k+1) for k in k_values}
    
    for k in k_values:
        for i in range(len(sequence)-k+1):
            observed_kmers[k].add(sequence[i:i+k])
            
    linguistic_complexity = {
        f'{k}-mer': len(observed_kmers[k]) / possible_kmers[k]
        for k in k_values
    }
    
    # Shannon entropy
    base_freqs = np.array([sequence.count(base) for base in 'ACGT']) / len(sequence)
    entropy = -np.sum(base_freqs * np.log2(base_freqs + 1e-10))
    
    return {
        'linguistic_complexity': linguistic_complexity,
        'shannon_entropy': entropy,
        'normalized_entropy': entropy / 2.0  # Max entropy for DNA is 2 bits
    }

def _find_sequence_motifs(sequence: str) -> Dict[str, List[Dict[str, Any]]]:
    """Find common DNA sequence motifs."""
    motifs = defaultdict(list)
    
    # Common regulatory motifs
    regulatory_patterns = {
        'TATA_box': r'TATA[AT]A[AT]',
        'CpG_island': r'CG',
        'CAAT_box': r'CCAAT',
        'GC_box': r'GGGCGG',
    }
    
    for motif_name, pattern in regulatory_patterns.items():
        for match in re.finditer(pattern, sequence):
            motifs['regulatory'].append({
                'type': motif_name,
                'start': match.start(),
                'end': match.end(),
                'sequence': match.group()
            })
            
    # Search for other common motifs
    common_motifs = ['AATAAA', 'GATA', 'CAGCTG']
    for motif in common_motifs:
        pos = 0
        while True:
            pos = sequence.find(motif, pos)
            if pos == -1:
                break
            motifs['common'].append({
                'sequence': motif,
                'position': pos
            })
            pos += 1
            
    return dict(motifs)

def _predict_secondary_structures(sequence: str) -> List[Dict[str, Any]]:
    """Predict potential DNA secondary structures."""
    structures = []
    
    # Find potential hairpins
    min_stem_length = 4
    max_loop_size = 8
    
    for i in range(len(sequence) - 2*min_stem_length):
        for stem_length in range(min_stem_length, min(10, (len(sequence)-i)//2)):
            for loop_size in range(3, max_loop_size+1):
                if i + 2*stem_length + loop_size > len(sequence):
                    continue
                    
                left_arm = sequence[i:i+stem_length]
                right_arm = sequence[i+stem_length+loop_size:i+2*stem_length+loop_size]
                
                if _are_complementary(left_arm, right_arm):
                    structures.append({
                        'type': 'hairpin',
                        'start': i,
                        'end': i + 2*stem_length + loop_size,
                        'stem_length': stem_length,
                        'loop_size': loop_size,
                        'stability': _calculate_stem_stability(left_arm, right_arm)
                    })
                    
    return structures

def _are_complementary(seq1: str, seq2: str) -> bool:
    """Check if two sequences are complementary."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    rev_seq2 = seq2[::-1]
    matches = sum(1 for b1, b2 in zip(seq1, rev_seq2) 
                 if b2 == complement.get(b1))
    return matches >= 0.8 * len(seq1)  # Allow some mismatches

def _calculate_stem_stability(arm1: str, arm2: str) -> float:
    """Calculate stability score for stem structure."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    rev_arm2 = arm2[::-1]
    
    # Base pairing energy contribution
    pair_energy = {
        ('G', 'C'): 3.0,
        ('C', 'G'): 3.0,
        ('A', 'T'): 2.0,
        ('T', 'A'): 2.0
    }
    
    energy = 0
    for b1, b2 in zip(arm1, rev_arm2):
        if b2 == complement.get(b1):
            energy += pair_energy.get((b1, b2), 0)
            
    return energy / (3.0 * len(arm1))  # Normalize to 0-1 range

def _analyze_repeats(sequence: str) -> Dict[str, List[Dict[str, Any]]]:
    """Analyze sequence repeats."""
    repeats = {
        'direct': [],
        'inverted': [],
        'tandem': []
    }
    
    # Find direct repeats
    min_repeat_len = 4
    max_repeat_len = 10
    
    for length in range(min_repeat_len, max_repeat_len + 1):
        for i in range(len(sequence) - length):
            pattern = sequence[i:i+length]
            if sequence.count(pattern) > 1:
                repeats['direct'].append({
                    'pattern': pattern,
                    'length': length,
                    'count': sequence.count(pattern),
                    'positions': [m.start() for m in re.finditer(pattern, sequence)]
                })
                
    # Find inverted repeats
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    for length in range(min_repeat_len, max_repeat_len + 1):
        for i in range(len(sequence) - length):
            pattern = sequence[i:i+length]
            rev_comp = ''.join(complement[b] for b in reversed(pattern))
            if rev_comp in sequence[i+length:]:
                repeats['inverted'].append({
                    'pattern': pattern,
                    'reverse_complement': rev_comp,
                    'length': length,
                    'position': i
                })
                
    # Find tandem repeats
    for length in range(2, 6):
        i = 0
        while i < len(sequence) - 2*length:
            pattern = sequence[i:i+length]
            repeat_count = 1
            current_pos = i + length
            
            while (current_pos + length <= len(sequence) and 
                   sequence[current_pos:current_pos+length] == pattern):
                repeat_count += 1
                current_pos += length
                
            if repeat_count > 1:
                repeats['tandem'].append({
                    'pattern': pattern,
                    'length': length,
                    'count': repeat_count,
                    'start': i,
                    'end': current_pos
                })
                i = current_pos
            else:
                i += 1
                
    return repeats

def _calculate_quality_metrics(sequence: str) -> Dict[str, float]:
    """Calculate sequence quality metrics."""
    # Base balance score
    base_counts = np.array([sequence.count(base) for base in 'ACGT'])
    base_freqs = base_counts / len(sequence)
    base_balance = 1 - np.std(base_freqs)
    
    # Local complexity score
    complexity_scores = []
    window_size = 10
    for i in range(0, len(sequence) - window_size + 1, window_size):
        window = sequence[i:i+window_size]
        unique_bases = len(set(window))
        complexity_scores.append(unique_bases / window_size)
    
    local_complexity = np.mean(complexity_scores) if complexity_scores else 0
    
    return {
        'base_balance': float(base_balance),
        'local_complexity': float(local_complexity),
        'sequence_quality': float((base_balance + local_complexity) / 2)
    }