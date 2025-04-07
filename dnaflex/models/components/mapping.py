"""Data mapping and transformation utilities."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import jax.numpy as jnp

class SequenceMapper:
    """Maps DNA/RNA sequences to numerical representations."""
    
    def __init__(self):
        """Initialize sequence mapper."""
        self.nucleotide_map = {
            'A': 0, 'T': 1, 'G': 2, 'C': 3,
            'U': 1,  # Map U (RNA) to same encoding as T
            'N': 4   # Unknown base
        }
        
        self.reverse_map = {v: k for k, v in self.nucleotide_map.items()}
        
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Convert sequence string to numerical encoding.
        
        Args:
            sequence: DNA/RNA sequence string
            
        Returns:
            Array of numerical encodings
        """
        return np.array([self.nucleotide_map.get(base.upper(), 4) 
                        for base in sequence])
                        
    def decode_sequence(self, encoding: np.ndarray) -> str:
        """Convert numerical encoding back to sequence string.
        
        Args:
            encoding: Array of numerical encodings
            
        Returns:
            DNA sequence string
        """
        return ''.join(self.reverse_map[idx] for idx in encoding)

class StructureMapper:
    """Maps 3D structural features to numerical representations."""
    
    def __init__(self, max_atoms: int = 34):
        """Initialize structure mapper.
        
        Args:
            max_atoms: Maximum number of atoms per nucleotide
        """
        self.max_atoms = max_atoms
        
    def encode_coordinates(self, 
                         coords: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode atomic coordinates with padding.
        
        Args:
            coords: [N, A, 3] array of atomic coordinates
                   where N=num residues, A=atoms per residue
            mask: Optional boolean mask of valid atoms
            
        Returns:
            Padded coordinate array
        """
        N = coords.shape[0]
        padded = np.zeros((N, self.max_atoms, 3))
        
        if mask is None:
            # Use all atoms up to max_atoms
            A = min(coords.shape[1], self.max_atoms)
            padded[:, :A] = coords[:, :A]
        else:
            # Only use masked atoms
            for i in range(N):
                valid_atoms = coords[i, mask[i]]
                padded[i, :len(valid_atoms)] = valid_atoms
                
        return padded
        
    def encode_atom_types(self,
                         atom_types: List[str],
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode atom types as one-hot vectors.
        
        Args:
            atom_types: List of atom type strings
            mask: Optional boolean mask of valid atoms
            
        Returns:
            One-hot encoded atom types
        """
        # Define standard atom types in nucleotides
        atom_map = {
            'P': 0, 'O': 1, 'C': 2, 'N': 3, 'H': 4,
            'OTHER': 5  # For any other atom types
        }
        
        N = len(atom_types)
        encoding = np.zeros((N, self.max_atoms, len(atom_map)))
        
        for i, atoms in enumerate(atom_types):
            if mask is not None:
                atoms = [a for j, a in enumerate(atoms) if mask[i, j]]
                
            for j, atom in enumerate(atoms[:self.max_atoms]):
                idx = atom_map.get(atom, atom_map['OTHER'])
                encoding[i, j, idx] = 1
                
        return encoding

class FeatureMapper:
    """Maps various molecular features to numerical representations."""
    
    def __init__(self):
        """Initialize feature mapper."""
        self.sequence_mapper = SequenceMapper()
        self.structure_mapper = StructureMapper()
        
    def encode_features(self,
                       sequence: str,
                       coords: Optional[np.ndarray] = None,
                       atom_types: Optional[List[str]] = None,
                       additional_features: Optional[Dict] = None) -> Dict:
        """Encode multiple feature types.
        
        Args:
            sequence: DNA/RNA sequence
            coords: Optional coordinate array
            atom_types: Optional atom type labels
            additional_features: Optional dict of additional features
            
        Returns:
            Dict of encoded features
        """
        features = {
            'sequence': self.sequence_mapper.encode_sequence(sequence)
        }
        
        if coords is not None:
            features['coordinates'] = self.structure_mapper.encode_coordinates(coords)
            
        if atom_types is not None:
            features['atom_types'] = self.structure_mapper.encode_atom_types(atom_types)
            
        if additional_features:
            features.update(additional_features)
            
        return features