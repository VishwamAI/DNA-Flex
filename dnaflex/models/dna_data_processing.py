"""DNA sequence and structure data processing."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dnaflex.common import base_config
from dnaflex.models.model_config import ModelConfig

class DNADataProcessor:
    """Process DNA sequence and structure data for model input."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._dna_vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self._complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        
    def process_sequence(self, sequence: str) -> Dict[str, np.ndarray]:
        """Process DNA sequence into model features.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Dictionary of processed features
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Basic validation
        if not all(base in self._dna_vocab for base in sequence):
            raise ValueError("Invalid DNA sequence")
            
        # Convert to indices
        seq_indices = np.array([self._dna_vocab[base] for base in sequence])
        
        # Create one-hot encoding
        seq_onehot = np.zeros((len(sequence), len(self._dna_vocab)))
        seq_onehot[np.arange(len(sequence)), seq_indices] = 1
        
        # Calculate sequence features
        gc_content = self._calculate_gc_content(sequence)
        kmers = self._extract_kmers(sequence)
        
        # Calculate positional features
        pos_features = self._calculate_positional_features(sequence)
        
        return {
            'sequence': seq_indices,
            'sequence_onehot': seq_onehot,
            'gc_content': gc_content,
            'kmers': kmers,
            'positional_features': pos_features
        }
        
    def process_structure(self, structure: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process DNA structure data.
        
        Args:
            structure: Dictionary containing structure information
            
        Returns:
            Processed structure features
        """
        # Extract atomic coordinates
        coords = self._extract_coordinates(structure)
        
        # Calculate structural features
        backbone_angles = self._calculate_backbone_angles(coords)
        local_structure = self._analyze_local_structure(coords)
        
        # Calculate distance matrices
        dist_matrix = self._calculate_distance_matrix(coords)
        contact_matrix = dist_matrix < 10.0  # Contact threshold of 10 Å
        
        return {
            'coordinates': coords,
            'backbone_angles': backbone_angles,
            'local_structure': local_structure,
            'distance_matrix': dist_matrix,
            'contact_matrix': contact_matrix
        }
        
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of sequence."""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
        
    def _extract_kmers(self, sequence: str, k: int = 3) -> np.ndarray:
        """Extract k-mer frequencies."""
        kmers = {}
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmers[kmer] = kmers.get(kmer, 0) + 1
            
        # Convert to frequency vector
        kmer_freqs = np.zeros(4**k)
        for kmer, count in kmers.items():
            idx = self._kmer_to_index(kmer)
            if idx is not None:
                kmer_freqs[idx] = count / (len(sequence) - k + 1)
                
        return kmer_freqs
        
    def _kmer_to_index(self, kmer: str) -> Optional[int]:
        """Convert k-mer to numerical index."""
        try:
            index = 0
            for i, base in enumerate(reversed(kmer)):
                index += self._dna_vocab[base] * (4**i)
            return index
        except KeyError:
            return None
            
    def _calculate_positional_features(self, sequence: str) -> np.ndarray:
        """Calculate position-specific features."""
        seq_len = len(sequence)
        features = np.zeros((seq_len, 4))
        
        # Relative position
        features[:, 0] = np.arange(seq_len) / seq_len
        
        # Local GC content
        window = 5
        for i in range(seq_len):
            start = max(0, i - window)
            end = min(seq_len, i + window + 1)
            window_seq = sequence[start:end]
            features[i, 1] = self._calculate_gc_content(window_seq)
            
        # Distance to nearest motif
        motifs = ['TATA', 'CCAAT', 'GGGCGG']
        for i in range(seq_len):
            min_dist = seq_len
            for motif in motifs:
                motif_len = len(motif)
                for j in range(max(0, i - 20), min(seq_len - motif_len + 1, i + 20)):
                    if sequence[j:j+motif_len] == motif:
                        dist = abs(i - j)
                        min_dist = min(min_dist, dist)
            features[i, 2] = min_dist / 20.0
            
        # Sequence complexity
        for i in range(seq_len):
            start = max(0, i - 5)
            end = min(seq_len, i + 6)
            window_seq = sequence[start:end]
            features[i, 3] = len(set(window_seq)) / len(window_seq)
            
        return features
        
    def _extract_coordinates(self, structure: Dict[str, Any]) -> np.ndarray:
        """Extract atomic coordinates from structure."""
        if 'coordinates' not in structure:
            raise ValueError("Structure must contain coordinates")
            
        return np.array(structure['coordinates'])
        
    def _calculate_backbone_angles(self, coords: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate DNA backbone angles."""
        if len(coords.shape) != 3 or coords.shape[2] != 3:
            raise ValueError("Invalid coordinate shape")
            
        num_residues = coords.shape[0]
        angles = {
            'alpha': np.zeros(num_residues-3),
            'beta': np.zeros(num_residues-3),
            'gamma': np.zeros(num_residues-3),
            'delta': np.zeros(num_residues-3),
            'epsilon': np.zeros(num_residues-3),
            'zeta': np.zeros(num_residues-3)
        }
        
        # Calculate backbone torsion angles
        for i in range(num_residues - 3):
            angles['alpha'][i] = self._calculate_torsion(
                coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3]
            )
            # Calculate other angles similarly
            
        return angles
        
    def _calculate_torsion(self, p1: np.ndarray, p2: np.ndarray, 
                         p3: np.ndarray, p4: np.ndarray) -> float:
        """Calculate torsion angle between four points."""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        m1 = np.cross(n1, b2/np.linalg.norm(b2))
        
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        return np.arctan2(y, x)
        
    def _analyze_local_structure(self, coords: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze local structural properties."""
        num_residues = coords.shape[0]
        properties = {
            'bend_angle': np.zeros(num_residues-2),
            'rise': np.zeros(num_residues-1),
            'twist': np.zeros(num_residues-1)
        }
        
        # Calculate local structural parameters
        for i in range(num_residues - 2):
            properties['bend_angle'][i] = self._calculate_bend_angle(
                coords[i], coords[i+1], coords[i+2]
            )
            
        for i in range(num_residues - 1):
            properties['rise'][i] = np.linalg.norm(coords[i+1] - coords[i])
            properties['twist'][i] = self._calculate_twist(coords[i], coords[i+1])
            
        return properties
        
    def _calculate_bend_angle(self, p1: np.ndarray, p2: np.ndarray, 
                            p3: np.ndarray) -> float:
        """Calculate bend angle between three points."""
        v1 = p2 - p1
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
    def _calculate_twist(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate twist angle between base pairs."""
        # Simplified twist calculation
        return np.arctan2(np.linalg.norm(np.cross(p1, p2)), np.dot(p1, p2))
        
    def _calculate_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Calculate pairwise distance matrix."""
        num_atoms = coords.shape[0]
        dist_matrix = np.zeros((num_atoms, num_atoms))
        
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
                
        return dist_matrix