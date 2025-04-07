"""Chemical component dictionary (CCD) related constants and helper functions."""

import dataclasses
from typing import Dict, Optional, List

@dataclasses.dataclass
class ChemicalComponent:
    """Represents a chemical component in the CCD.

    Attributes:
        id: Component ID (e.g., 'ALA', 'ATP')
        type: Component type (e.g., 'L-peptide', 'RNA')
        name: Full chemical name
        formula: Chemical formula
        systematic_name: Systematic chemical name
        pdbx_smiles: SMILES representation
        atoms: List of atom names in this component
        elements: List of element types for each atom
        bonds: List of bonds between atoms
    """
    id: str
    type: str 
    name: str
    formula: str
    systematic_name: Optional[str] = None
    pdbx_smiles: Optional[str] = None
    atoms: Optional[List[str]] = None
    elements: Optional[List[str]] = None
    bonds: Optional[List[tuple[str, str, str]]] = None

class Ccd:
    """Chemical Component Dictionary.
    
    A minimal implementation that stores chemical component information.
    """
    def __init__(self):
        self._components: Dict[str, ChemicalComponent] = {}

    def add_component(self, comp: ChemicalComponent):
        """Add a chemical component."""
        self._components[comp.id] = comp

    def get(self, comp_id: str) -> Optional[ChemicalComponent]:
        """Get a chemical component by ID."""
        return self._components.get(comp_id)

    def __contains__(self, comp_id: str) -> bool:
        """Check if component ID exists in dictionary."""
        return comp_id in self._components

    def __iter__(self):
        """Iterate over component IDs."""
        return iter(self._components)

    def __len__(self) -> int:
        """Get number of components."""
        return len(self._components)

# Standard component types
COMP_TYPES = {
    'L-PEPTIDE LINKING': 'L-peptide',
    'D-PEPTIDE LINKING': 'D-peptide', 
    'RNA LINKING': 'RNA',
    'DNA LINKING': 'DNA',
    'SACCHARIDE': 'saccharide',
    'NON-POLYMER': 'non-polymer',
    'OTHER': 'other'
}

def get_component_type(ccd_type: str) -> str:
    """Convert CCD component type to standard type."""
    return COMP_TYPES.get(ccd_type, 'other')

def create_standard_component(
    comp_id: str,
    comp_type: str,
    name: str,
    formula: str,
    atoms: Optional[List[str]] = None,
    elements: Optional[List[str]] = None
) -> ChemicalComponent:
    """Create a standard chemical component with basic information."""
    return ChemicalComponent(
        id=comp_id,
        type=get_component_type(comp_type),
        name=name,
        formula=formula,
        atoms=atoms,
        elements=elements
    )

# Modified DNA bases
MODIFIED_DNA_BASES = {
    '5MC': {  # 5-methylcytosine
        'name': '5-methylcytosine',
        'parent': 'DC',
        'formula': 'C5H7N3O',
        'atoms': [
            'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C5M', 'C6'  # C5M is methyl group
        ]
    },
    'M2G': {  # N2-methylguanine
        'name': 'N2-methylguanine',
        'parent': 'DG',
        'formula': 'C6H7N5O',
        'atoms': [
            'N1', 'C2', 'N2', 'CN2', 'N3', 'C4', 'C5', 'C6', 'O6', 'N7', 'C8', 'N9'  # CN2 is methyl
        ]
    },
    'DHU': {  # Dihydrouracil
        'name': 'Dihydrouracil',
        'parent': 'DT',
        'formula': 'C4H6N2O2',
        'atoms': [
            'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'
        ]
    }
}