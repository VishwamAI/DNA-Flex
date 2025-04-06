"""Core DNA molecule structure and handling functionality."""

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Iterator, Any
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

@dataclasses.dataclass
class Structure:
    """Container for structure data."""
    
    name: str
    atom_name: List[str]
    atom_element: List[str] 
    atom_x: List[float]
    atom_y: List[float]
    atom_z: List[float]
    atom_b_factor: List[float]
    res_name: List[str]
    res_id: List[int]
    chain_id: List[str]
    chain_type: List[str]
    all_residues: Dict[str, List[tuple[str, int]]]
    chemical_components_data: Optional[Any] = None

    def __post_init__(self):
        """Convert lists to numpy arrays."""
        self.atom_name = np.array(self.atom_name, dtype=object)
        self.atom_element = np.array(self.atom_element, dtype=object)
        self.atom_x = np.array(self.atom_x, dtype=float)
        self.atom_y = np.array(self.atom_y, dtype=float)
        self.atom_z = np.array(self.atom_z, dtype=float)
        self.atom_b_factor = np.array(self.atom_b_factor, dtype=float)
        self.res_name = np.array(self.res_name, dtype=object)
        self.res_id = np.array(self.res_id, dtype=int)
        self.chain_id = np.array(self.chain_id, dtype=object)
        self.chain_type = np.array(self.chain_type, dtype=object)

    def iter_atoms(self) -> Iterator[Dict[str, Any]]:
        """Iterate over atoms in the structure."""
        for i in range(len(self.atom_name)):
            yield {
                'atom_name': self.atom_name[i],
                'atom_element': self.atom_element[i],
                'atom_x': self.atom_x[i],
                'atom_y': self.atom_y[i],
                'atom_z': self.atom_z[i],
                'atom_b_factor': self.atom_b_factor[i],
                'res_name': self.res_name[i],
                'res_id': self.res_id[i],
                'chain_id': self.chain_id[i],
                'chain_type': self.chain_type[i]
            }

    def iter_residues(self) -> Iterator[Dict[str, Any]]:
        """Iterate over residues in the structure."""
        seen = set()
        for i in range(len(self.res_id)):
            key = (self.chain_id[i], self.res_id[i])
            if key not in seen:
                seen.add(key)
                yield {
                    'res_name': self.res_name[i],
                    'res_id': self.res_id[i],
                    'chain_id': self.chain_id[i],
                    'chain_type': self.chain_type[i]
                }

    def iter_chains(self) -> Iterator[Dict[str, str]]:
        """Iterate over chains in the structure."""
        seen = set()
        for i in range(len(self.chain_id)):
            if self.chain_id[i] not in seen:
                seen.add(self.chain_id[i])
                yield {
                    'chain_id': self.chain_id[i],
                    'chain_type': self.chain_type[i]
                }

    def chain_res_name_sequence(
        self,
        include_missing_residues: bool = True,
        fix_non_standard_polymer_res: bool = False
    ) -> Dict[str, List[str]]:
        """Get residue name sequence for each chain."""
        from dnaflex.constants import mmcif_names
        
        sequences = {}
        
        for chain in self.iter_chains():
            chain_id = chain['chain_id']
            if include_missing_residues:
                sequences[chain_id] = [
                    name for name, _ in self.all_residues[chain_id]
                ]
            else:
                res_names = []
                seen = set()
                for i in range(len(self.res_id)):
                    if (self.chain_id[i] == chain_id and 
                        self.res_id[i] not in seen):
                        seen.add(self.res_id[i])
                        res_names.append(self.res_name[i])
                sequences[chain_id] = res_names

            if fix_non_standard_polymer_res:
                for i, res_name in enumerate(sequences[chain_id]):
                    if mmcif_names.is_standard_polymer_type(chain['chain_type']):
                        sequences[chain_id][i] = mmcif_names.fix_non_standard_polymer_res(
                            res_name=res_name,
                            chain_type=chain['chain_type']
                        )

        return sequences

def from_atom_arrays(
    name: str,
    all_residues: Dict[str, List[tuple[str, int]]],
    chain_id: List[str],
    chain_type: List[str],
    res_id: List[int],
    res_name: List[str],
    atom_name: List[str],
    atom_element: List[str],
    atom_x: List[float],
    atom_y: List[float],
    atom_z: List[float],
    atom_b_factor: List[float],
) -> Structure:
    """Create a Structure from atom-level arrays."""
    return Structure(
        name=name,
        all_residues=all_residues,
        chain_id=chain_id,
        chain_type=chain_type,
        res_id=res_id,
        res_name=res_name,
        atom_name=atom_name,
        atom_element=atom_element,
        atom_x=atom_x,
        atom_y=atom_y,
        atom_z=atom_z,
        atom_b_factor=atom_b_factor
    )