"""DNA flexibility analysis and prediction module."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .structure import DnaStructure, DnaChain, DnaResidue

class FlexibilityAnalyzer:
    """Analyzes DNA flexibility based on structure and sequence."""
    
    # Base stacking energy parameters (approximate values in kcal/mol)
    STACKING_ENERGIES = {
        ('A', 'A'): -5.37, ('A', 'T'): -6.57, ('A', 'C'): -5.27, ('A', 'G'): -6.78,
        ('T', 'A'): -6.57, ('T', 'T'): -5.37, ('T', 'C'): -5.37, ('T', 'G'): -5.27,
        ('C', 'A'): -5.27, ('C', 'T'): -5.37, ('C', 'C'): -5.37, ('C', 'G'): -8.26,
        ('G', 'A'): -6.78, ('G', 'T'): -5.27, ('G', 'C'): -8.26, ('G', 'G'): -5.37
    }
    
    # B-DNA parameters
    BDNA_RISE = 3.4  # Å per base pair
    BDNA_TWIST = 36.0  # degrees per base pair
    
    def __init__(self, structure: DnaStructure):
        self.structure = structure
        
    def calculate_base_step_parameters(self, chain: DnaChain) -> Dict[str, np.ndarray]:
        """Calculate base step parameters (rise, roll, twist) between consecutive base pairs."""
        parameters = {
            'rise': [],
            'roll': [],
            'twist': []
        }
        
        residues = sorted(chain._residues.items())
        for i in range(len(residues) - 1):
            curr_res = residues[i][1]
            next_res = residues[i+1][1]
            
            # Calculate rise using C1' atoms
            if "C1'" in curr_res.atoms and "C1'" in next_res.atoms:
                c1_curr = np.array([curr_res.atoms["C1'"].x, 
                                  curr_res.atoms["C1'"].y,
                                  curr_res.atoms["C1'"].z])
                c1_next = np.array([next_res.atoms["C1'"].x,
                                  next_res.atoms["C1'"].y,
                                  next_res.atoms["C1'"].z])
                rise = np.linalg.norm(c1_next - c1_curr)
                parameters['rise'].append(rise)
                
                # Calculate approximate roll and twist using backbone atoms
                if all(atom in curr_res.atoms and atom in next_res.atoms 
                      for atom in ["P", "O3'", "C3'"]):
                    curr_plane = self._calculate_base_plane(curr_res)
                    next_plane = self._calculate_base_plane(next_res)
                    
                    if curr_plane is not None and next_plane is not None:
                        roll = self._calculate_angle(curr_plane[1], next_plane[1])
                        twist = self._calculate_angle(curr_plane[0], next_plane[0])
                        
                        parameters['roll'].append(roll)
                        parameters['twist'].append(twist)
                    
        return {k: np.array(v) for k, v in parameters.items()}
    
    def predict_flexibility(self, chain: DnaChain) -> np.ndarray:
        """Predict flexibility scores for each base in the chain."""
        sequence = chain.sequence
        flexibility_scores = np.zeros(len(sequence))
        
        # Calculate base stacking contributions
        for i in range(len(sequence)-1):
            curr_base = sequence[i]
            next_base = sequence[i+1]
            pair = (curr_base, next_base)
            
            # Get stacking energy
            energy = self.STACKING_ENERGIES.get(pair, -5.0)  # default if unknown
            
            # Convert energy to flexibility score (higher energy = less flexible)
            # Normalize to 0-1 range where 1 is most flexible
            flex_score = 1.0 - (abs(energy) / 10.0)  # 10.0 is approximate max energy
            
            # Influence both current and next base
            flexibility_scores[i] += flex_score * 0.7
            flexibility_scores[i+1] += flex_score * 0.3
            
        # Adjust for sequence-dependent effects
        for i in range(len(sequence)):
            # AT-rich regions are generally more flexible
            if sequence[i] in 'AT':
                flexibility_scores[i] *= 1.2
                
        # Normalize final scores
        flexibility_scores = np.clip(flexibility_scores, 0, 1)
        
        return flexibility_scores
    
    def _calculate_base_plane(self, residue: DnaResidue) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Calculate base plane normal vector and reference vector for a residue."""
        if all(atom in residue.atoms for atom in ["C1'", "N1", "C4"]):
            c1 = np.array([residue.atoms["C1'"].x, residue.atoms["C1'"].y, residue.atoms["C1'"].z])
            n1 = np.array([residue.atoms["N1"].x, residue.atoms["N1"].y, residue.atoms["N1"].z])
            c4 = np.array([residue.atoms["C4"].x, residue.atoms["C4"].y, residue.atoms["C4"].z])
            
            # Calculate vectors defining base plane
            v1 = n1 - c1
            v2 = c4 - c1
            
            # Normal vector to base plane
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            # Reference vector for measuring twist
            ref = v1 / np.linalg.norm(v1)
            
            return ref, normal
            
        return None
    
    @staticmethod
    def _calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        return np.degrees(np.arccos(cos_angle))
        
    def identify_flexible_regions(self, chain: DnaChain, 
                                window_size: int = 4,
                                threshold: float = 0.6) -> List[Tuple[int, int]]:
        """Identify continuous regions of high flexibility."""
        flexibility = self.predict_flexibility(chain)
        
        # Use sliding window to find regions of sustained flexibility
        flexible_regions = []
        start_idx = None
        
        for i in range(len(flexibility) - window_size + 1):
            window_avg = np.mean(flexibility[i:i+window_size])
            
            if window_avg > threshold:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                flexible_regions.append((start_idx, i))
                start_idx = None
                
        # Handle case where flexible region extends to end
        if start_idx is not None:
            flexible_regions.append((start_idx, len(flexibility)))
            
        return flexible_regions