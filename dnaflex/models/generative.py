"""DNA sequence generation and variation module."""

from typing import Dict, List, Any
import numpy as np

class DnaGenerator:
    """Generate DNA sequence variations."""
    
    def __init__(self):
        self.mutation_rate = 0.01
        self.indel_rate = 0.005
        
    def generate(self, sequence: str, num_variants: int = 5) -> Dict[str, Any]:
        """Generate variations of input sequence."""
        variants = []
        scores = []
        
        for _ in range(num_variants):
            # Generate variant
            variant = self._create_variant(sequence)
            variants.append(variant)
            
            # Score variant
            score = self._score_variant(variant, sequence)
            scores.append(score)
            
        return {
            'variants': variants,
            'scores': scores,
            'original': sequence
        }
        
    def _create_variant(self, sequence: str) -> str:
        """Create a single sequence variant."""
        bases = list(sequence)
        
        # Apply point mutations
        for i in range(len(bases)):
            if np.random.random() < self.mutation_rate:
                bases[i] = np.random.choice(['A', 'T', 'G', 'C'])
                
        # Apply insertions/deletions
        for i in range(len(bases)):
            if np.random.random() < self.indel_rate:
                if np.random.random() < 0.5:  # Insertion
                    bases.insert(i, np.random.choice(['A', 'T', 'G', 'C']))
                elif len(bases) > 1:  # Deletion (keep at least 1 base)
                    bases.pop(i)
                    
        return ''.join(bases)
        
    def _score_variant(self, variant: str, original: str) -> float:
        """Score a variant based on similarity to original."""
        # Simple scoring based on sequence similarity and structural properties
        similarity = self._sequence_similarity(variant, original)
        stability = self._estimate_stability(variant)
        
        return 0.7 * similarity + 0.3 * stability
        
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity score."""
        # Use Levenshtein distance for sequence similarity
        max_len = max(len(seq1), len(seq2))
        changes = sum(1 for i in range(min(len(seq1), len(seq2))) if seq1[i] != seq2[i])
        changes += abs(len(seq1) - len(seq2))
        
        return 1.0 - (changes / max_len)
        
    def _estimate_stability(self, sequence: str) -> float:
        """Estimate sequence stability."""
        # Simple stability estimation based on GC content and repeats
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        
        # Penalize long repeat sequences
        repeat_penalty = 0
        for i in range(len(sequence)-3):
            if sequence[i:i+3] in sequence[i+3:]:
                repeat_penalty += 0.1
                
        stability = (0.5 * gc_content + 0.5) * (1 - min(repeat_penalty, 0.5))
        return np.clip(stability, 0, 1)

# Create global instance
dna_generation = DnaGenerator()