"""Input data handling for molecular structure folding."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from dnaflex.constants import chemical_components
from dnaflex.structure import structure

@dataclass
class Input:
    """Container for folding input data."""
    
    name: str
    sequence: str
    structure: Optional[Structure] = None
    features: Optional[Dict[str, List[float]]] = None
    
    def to_structure(self, ccd: chemical_components.Ccd) -> Structure:
        """Convert input to Structure object."""
        if self.structure is not None:
            return self.structure
            
        # Create structure from sequence
        struc = structure.Structure.from_sequence(
            sequence=self.sequence,
            ccd=ccd,
            name=self.name
        )
        
        # Add features if available
        if self.features:
            struc = struc.copy_and_update_features(self.features)
            
        return struc
        
    @classmethod
    def from_pdb(cls, pdb_path: str, name: Optional[str] = None) -> 'Input':
        """Create input from PDB file."""
        from dnaflex.parsers.parser import DnaParser
        
        parser = DnaParser()
        struc = parser.parse_pdb(pdb_path)
        return cls(
            name=name or pdb_path,
            sequence=struc.get_sequence(),
            structure=struc
        )