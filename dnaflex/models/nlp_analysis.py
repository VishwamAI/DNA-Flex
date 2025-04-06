"""DNA sequence pattern analysis using NLP techniques."""

from typing import Dict, List, Any
import numpy as np
from collections import defaultdict

class SequenceNLP:
    """NLP-based DNA sequence analysis."""
    
    def __init__(self):
        # K-mer sizes for pattern analysis
        self.kmer_sizes = [2, 3, 4, 5]
        
        # Sequence motifs of interest
        self.functional_motifs = {
            'promoter': ['TATA', 'CAAT', 'GC'],
            'enhancer': ['CCAAT', 'GGGCGG'],
            'terminator': ['AATAAA'],
            'splice': ['GT', 'AG']
        }
        
    def analyze(self, sequence: str) -> Dict[str, Any]:
        """Analyze DNA sequence patterns."""
        # Perform k-mer analysis
        kmer_patterns = self._analyze_kmers(sequence)
        
        # Find sequence motifs
        motifs = self._find_motifs(sequence)
        
        # Analyze sequence complexity
        complexity = self._analyze_complexity(sequence)
        
        # Identify repetitive elements
        repeats = self._find_repeats(sequence)
        
        return {
            'kmer_patterns': kmer_patterns,
            'motifs': motifs,
            'complexity': complexity,
            'repeats': repeats,
            'sequence_features': self._extract_sequence_features(sequence)
        }
        
    def _analyze_kmers(self, sequence: str) -> Dict[str, Any]:
        """Analyze k-mer frequencies and patterns."""
        kmer_analysis = {}
        
        for k in self.kmer_sizes:
            kmers = defaultdict(int)
            # Count k-mers
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i+k]
                kmers[kmer] += 1
                
            # Calculate statistics
            total_kmers = len(sequence) - k + 1
            frequencies = {
                kmer: count / total_kmers 
                for kmer, count in kmers.items()
            }
            
            # Find significant patterns
            significant = {
                kmer: freq for kmer, freq in frequencies.items()
                if freq > 1.5 * (1 / 4**k)  # Higher than random expectation
            }
            
            kmer_analysis[f'{k}-mers'] = {
                'frequencies': frequencies,
                'significant_patterns': significant,
                'diversity': len(kmers) / min(4**k, total_kmers)
            }
            
        return kmer_analysis
        
    def _find_motifs(self, sequence: str) -> Dict[str, List[Dict[str, Any]]]:
        """Find functional sequence motifs."""
        motif_findings = defaultdict(list)
        
        for motif_type, patterns in self.functional_motifs.items():
            for pattern in patterns:
                pos = 0
                while True:
                    pos = sequence.find(pattern, pos)
                    if pos == -1:
                        break
                        
                    # Score motif match
                    score = self._score_motif_match(sequence, pos, pattern)
                    
                    motif_findings[motif_type].append({
                        'pattern': pattern,
                        'position': pos,
                        'score': score,
                        'context': sequence[max(0, pos-5):pos+len(pattern)+5]
                    })
                    pos += 1
                    
        return dict(motif_findings)
        
    def _score_motif_match(self, sequence: str, position: int, motif: str) -> float:
        """Score a motif match based on context."""
        # Consider local sequence composition
        context_start = max(0, position - 5)
        context_end = min(len(sequence), position + len(motif) + 5)
        context = sequence[context_start:context_end]
        
        # Base score
        score = 1.0
        
        # Adjust for GC content in context
        gc_content = (context.count('G') + context.count('C')) / len(context)
        if 0.4 <= gc_content <= 0.6:  # Optimal GC range
            score *= 1.2
            
        # Penalize if in repetitive region
        if self._is_repetitive(context):
            score *= 0.8
            
        return score
        
    def _analyze_complexity(self, sequence: str) -> Dict[str, float]:
        """Analyze sequence complexity measures."""
        # Calculate linguistic complexity
        observed_kmers = 0
        possible_kmers = 0
        
        for k in range(1, min(6, len(sequence))):
            kmers = set()
            for i in range(len(sequence) - k + 1):
                kmers.add(sequence[i:i+k])
            observed_kmers += len(kmers)
            possible_kmers += min(4**k, len(sequence) - k + 1)
            
        linguistic_complexity = observed_kmers / possible_kmers if possible_kmers > 0 else 0
        
        # Calculate entropy
        base_freqs = {
            base: sequence.count(base) / len(sequence)
            for base in 'ACGT'
        }
        entropy = -sum(
            freq * np.log2(freq) for freq in base_freqs.values() if freq > 0
        )
        
        return {
            'linguistic_complexity': linguistic_complexity,
            'entropy': entropy,
            'normalized_entropy': entropy / 2.0  # Max entropy for DNA is 2 bits
        }
        
    def _find_repeats(self, sequence: str) -> List[Dict[str, Any]]:
        """Identify repetitive elements in sequence."""
        repeats = []
        min_repeat_len = 3
        max_repeat_len = 10
        
        # Search for tandem repeats
        for length in range(min_repeat_len, max_repeat_len + 1):
            for i in range(len(sequence) - length):
                pattern = sequence[i:i+length]
                if pattern in sequence[i+length:]:
                    # Find all occurrences
                    occurrences = []
                    pos = 0
                    while True:
                        pos = sequence.find(pattern, pos)
                        if pos == -1:
                            break
                        occurrences.append(pos)
                        pos += 1
                        
                    if len(occurrences) > 1:
                        repeats.append({
                            'pattern': pattern,
                            'length': length,
                            'occurrences': occurrences,
                            'count': len(occurrences)
                        })
                        
        return repeats
        
    def _is_repetitive(self, sequence: str) -> bool:
        """Check if a sequence region is repetitive."""
        # Look for dinucleotide repeats
        for i in range(len(sequence)-3):
            if sequence[i:i+2] == sequence[i+2:i+4]:
                return True
        return False
        
    def _extract_sequence_features(self, sequence: str) -> Dict[str, Any]:
        """Extract additional sequence features."""
        features = {
            'length': len(sequence),
            'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence),
            'gc_skew': (sequence.count('G') - sequence.count('C')) / 
                      (sequence.count('G') + sequence.count('C') + 1e-6),
            'at_skew': (sequence.count('A') - sequence.count('T')) /
                      (sequence.count('A') + sequence.count('T') + 1e-6),
            'base_runs': self._find_base_runs(sequence)
        }
        
        return features
        
    def _find_base_runs(self, sequence: str) -> Dict[str, List[int]]:
        """Find runs of consecutive bases."""
        runs = defaultdict(list)
        current_base = sequence[0]
        current_run = 1
        
        for base in sequence[1:]:
            if base == current_base:
                current_run += 1
            else:
                if current_run > 1:
                    runs[current_base].append(current_run)
                current_base = base
                current_run = 1
                
        if current_run > 1:
            runs[current_base].append(current_run)
            
        return dict(runs)

# Create global instance
sequence_nlp = SequenceNLP()