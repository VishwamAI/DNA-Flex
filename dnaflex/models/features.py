"""DNA sequence feature extraction module."""

import numpy as np
from typing import Dict, List, Any
import re

class FeatureExtractor:
    """Extract features from DNA sequences for model input."""
    
    def __init__(self):
        # Initialize feature parameters
        self.kmer_sizes = [2, 3, 4]  # Sizes of k-mers to extract
        self.window_size = 10  # Size of sliding window for local features
        
    def extract_features(self, sequence: str) -> Dict[str, np.ndarray]:
        """Extract comprehensive feature set from DNA sequence.
        
        Args:
            sequence: Input DNA sequence
            
        Returns:
            Dictionary containing feature arrays
        """
        features = {
            'sequence_encoding': self._encode_sequence(sequence),
            'kmer_frequencies': self._extract_kmer_features(sequence),
            'structural_features': self._extract_structural_features(sequence),
            'complexity_features': self._extract_complexity_features(sequence),
            'local_features': self._extract_local_features(sequence)
        }
        
        return features
        
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """One-hot encode DNA sequence."""
        encoding_map = {'A': [1,0,0,0], 'C': [0,1,0,0], 
                       'G': [0,0,1,0], 'T': [0,0,0,1]}
        return np.array([encoding_map[base] for base in sequence])
        
    def _extract_kmer_features(self, sequence: str) -> np.ndarray:
        """Extract k-mer frequency features."""
        kmer_freqs = []
        
        for k in self.kmer_sizes:
            freq_dict = {}
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i+k]
                freq_dict[kmer] = freq_dict.get(kmer, 0) + 1
                
            # Normalize frequencies
            total = sum(freq_dict.values())
            kmer_freqs.extend([freq_dict.get(kmer, 0)/total 
                             for kmer in self._generate_kmers(k)])
                             
        return np.array(kmer_freqs)
        
    def _generate_kmers(self, k: int) -> List[str]:
        """Generate all possible k-mers."""
        if k == 1:
            return ['A', 'C', 'G', 'T']
        return [base + kmer for base in 'ACGT' 
                for kmer in self._generate_kmers(k-1)]
        
    def _extract_structural_features(self, sequence: str) -> np.ndarray:
        """Extract DNA structural features."""
        # Stability parameters
        stability_map = {
            'AA': 1.0, 'AT': 0.8, 'AG': 0.9, 'AC': 0.8,
            'TA': 0.6, 'TT': 1.0, 'TG': 0.7, 'TC': 0.7,
            'GA': 0.9, 'GT': 0.7, 'GG': 1.2, 'GC': 1.1,
            'CA': 0.8, 'CT': 0.7, 'CG': 1.1, 'CC': 1.2
        }
        
        # Calculate structural features
        features = []
        
        # Local stability
        stability_scores = []
        for i in range(len(sequence)-1):
            dinuc = sequence[i:i+2]
            stability_scores.append(stability_map.get(dinuc, 0.5))
        features.extend(self._summarize_local_scores(stability_scores))
        
        # GC content in windows
        gc_scores = []
        for i in range(0, len(sequence), self.window_size):
            window = sequence[i:i+self.window_size]
            gc_scores.append((window.count('G') + window.count('C')) / len(window))
        features.extend(self._summarize_local_scores(gc_scores))
        
        return np.array(features)
        
    def _extract_complexity_features(self, sequence: str) -> np.ndarray:
        """Extract sequence complexity features."""
        features = []
        
        # Linguistic complexity
        for k in range(1, 4):
            observed = len(set(sequence[i:i+k] 
                            for i in range(len(sequence)-k+1)))
            possible = min(4**k, len(sequence)-k+1)
            features.append(observed / possible)
            
        # Local complexity
        complexity_scores = []
        for i in range(0, len(sequence), self.window_size):
            window = sequence[i:i+self.window_size]
            unique_bases = len(set(window))
            complexity_scores.append(unique_bases / len(window))
        features.extend(self._summarize_local_scores(complexity_scores))
        
        return np.array(features)
        
    def _extract_local_features(self, sequence: str) -> np.ndarray:
        """Extract features using sliding windows."""
        local_features = []
        
        for i in range(0, len(sequence) - self.window_size + 1):
            window = sequence[i:i+self.window_size]
            
            # Base composition
            base_freqs = [window.count(base)/self.window_size 
                         for base in 'ACGT']
            local_features.extend(base_freqs)
            
            # Local complexity
            unique_bases = len(set(window))
            local_features.append(unique_bases / self.window_size)
            
            # Dinucleotide properties
            for dinuc in ['AA', 'AT', 'AG', 'AC', 'TA', 'GC']:
                count = sum(1 for i in range(len(window)-1) 
                          if window[i:i+2] == dinuc)
                local_features.append(count / (len(window)-1))
                
        return np.array(local_features)
        
    def _summarize_local_scores(self, scores: List[float]) -> List[float]:
        """Compute summary statistics for local score windows."""
        if not scores:
            return [0.0] * 4
            
        return [
            np.mean(scores),
            np.std(scores),
            np.min(scores),
            np.max(scores)
        ]