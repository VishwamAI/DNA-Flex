"""Core DNA molecule structure and handling functionality."""

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

@dataclasses.dataclass(frozen=True)
class DnaAtom:
    """Represents an atom in a DNA molecule."""
    id: str
    name: str  # e.g. 'P', 'O5\'', 'C3\'' etc
    element: str  # e.g. 'P', 'O', 'C', etc
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_factor: float = 0.0

@dataclasses.dataclass(frozen=True)
class DnaResidue:
    """Represents a DNA residue (nucleotide)."""
    name: str  # e.g. 'DA', 'DG', 'DC', 'DT'
    number: int
    atoms: Dict[str, DnaAtom]
    chain_id: str
    
    @property
    def base_type(self) -> str:
        """Returns the single letter code for the base."""
        base_map = {'DA': 'A', 'DG': 'G', 'DC': 'C', 'DT': 'T', 'DN': 'N'}
        return base_map.get(self.name, 'N')

class DnaChain:
    """Represents a DNA chain (strand)."""
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self._residues: Dict[int, DnaResidue] = {}
        
    def add_residue(self, residue: DnaResidue) -> None:
        """Add a residue to the chain."""
        self._residues[residue.number] = residue
        
    @property
    def sequence(self) -> str:
        """Get the DNA sequence as a string."""
        sorted_residues = sorted(self._residues.items())
        return ''.join(res.base_type for _, res in sorted_residues)

    def get_residue(self, number: int) -> Optional[DnaResidue]:
        """Get a residue by its number."""
        return self._residues.get(number)
        
    def __len__(self) -> int:
        return len(self._residues)

class DnaStructure:
    """Main class for handling DNA molecule structure."""
    def __init__(self):
        self._chains: Dict[str, DnaChain] = {}
        
    def add_chain(self, chain: DnaChain) -> None:
        """Add a chain to the structure."""
        self._chains[chain.chain_id] = chain
        
    def get_chain(self, chain_id: str) -> Optional[DnaChain]:
        """Get a chain by its ID."""
        return self._chains.get(chain_id)
        
    @property
    def chains(self) -> List[DnaChain]:
        """Get all chains in the structure."""
        return list(self._chains.values())
        
    def calculate_center_of_mass(self) -> np.ndarray:
        """Calculate the center of mass of the structure."""
        all_coords = []
        all_weights = []
        
        for chain in self.chains:
            for residue in chain._residues.values():
                for atom in residue.atoms.values():
                    all_coords.append([atom.x, atom.y, atom.z])
                    # Approximate atomic weights
                    weight_map = {'C': 12.0, 'N': 14.0, 'O': 16.0, 'P': 31.0}
                    all_weights.append(weight_map.get(atom.element, 1.0))
                    
        coords = np.array(all_coords)
        weights = np.array(all_weights)
        return np.average(coords, axis=0, weights=weights)

    def calculate_radius_of_gyration(self) -> float:
        """Calculate the radius of gyration of the structure."""
        com = self.calculate_center_of_mass()
        total_mass = 0.0
        weighted_r2 = 0.0
        
        for chain in self.chains:
            for residue in chain._residues.values():
                for atom in residue.atoms.values():
                    weight_map = {'C': 12.0, 'N': 14.0, 'O': 16.0, 'P': 31.0}
                    mass = weight_map.get(atom.element, 1.0)
                    coords = np.array([atom.x, atom.y, atom.z])
                    r2 = np.sum((coords - com) ** 2)
                    weighted_r2 += mass * r2
                    total_mass += mass
                    
        return np.sqrt(weighted_r2 / total_mass) if total_mass > 0 else 0.0