"""Functions for working with chemical components data."""

from collections import defaultdict
from typing import Dict, List, Optional

from dnaflex.constants import chemical_components as chem_comps

def get_all_atoms_in_entry(ccd: chem_comps.Ccd, res_name: str) -> Dict[str, List[str]]:
    """Get all atoms from a chemical component dictionary entry.
    
    Args:
        ccd: Chemical Component Dictionary
        res_name: Residue name to look up
        
    Returns:
        Dictionary with atom information keyed by mmCIF field names
    """
    comp = ccd.get(res_name)
    if comp is None:
        raise ValueError(f"Component {res_name} not found in CCD")
        
    result = defaultdict(list)
    
    if comp.atoms:
        for atom in comp.atoms:
            result['_chem_comp_atom.atom_id'].append(atom)
            result['_chem_comp_atom.type_symbol'].append(
                # Extract element from atom name (first character)
                atom[0] if atom[0].isalpha() else 'C'
            )
            
    return dict(result)

def get_atom_element(
    ccd: chem_comps.Ccd,
    res_name: str,
    atom_name: str
) -> Optional[str]:
    """Get the element type for a specific atom in a residue.
    
    Args:
        ccd: Chemical Component Dictionary
        res_name: Residue name
        atom_name: Atom name
        
    Returns:
        Element symbol or None if not found
    """
    comp = ccd.get(res_name)
    if comp is None or comp.atoms is None:
        return None
        
    # First look for exact match
    for a in comp.atoms:
        if a == atom_name:
            return atom_name[0] if atom_name[0].isalpha() else 'C'
            
    # Try case-insensitive match
    atom_name_lower = atom_name.lower()
    for a in comp.atoms:
        if a.lower() == atom_name_lower:
            return a[0] if a[0].isalpha() else 'C'
            
    return None

def get_bonds(
    ccd: chem_comps.Ccd,
    res_name: str
) -> List[tuple[str, str, str]]:
    """Get all bonds for a residue from the CCD.
    
    Args:
        ccd: Chemical Component Dictionary
        res_name: Residue name
        
    Returns:
        List of (atom1, atom2, bond_type) tuples
    """
    comp = ccd.get(res_name)
    if comp is None or comp.bonds is None:
        return []
        
    return comp.bonds

def get_canonical_name(ccd: chem_comps.Ccd, res_name: str) -> Optional[str]:
    """Get canonical name for a residue from the CCD.
    
    Args:
        ccd: Chemical Component Dictionary
        res_name: Residue name to look up
        
    Returns:
        Canonical name or None if not found
    """
    comp = ccd.get(res_name)
    if comp is None:
        return None
    return comp.name

def get_systematic_name(ccd: chem_comps.Ccd, res_name: str) -> Optional[str]:
    """Get systematic name for a residue from the CCD.
    
    Args:
        ccd: Chemical Component Dictionary
        res_name: Residue name to look up
        
    Returns:
        Systematic name or None if not found
    """
    comp = ccd.get(res_name)
    if comp is None:
        return None
    return comp.systematic_name

def get_component_type(ccd: chem_comps.Ccd, res_name: str) -> Optional[str]:
    """Get type of a chemical component.
    
    Args:
        ccd: Chemical Component Dictionary
        res_name: Residue name to look up
        
    Returns:
        Component type or None if not found
    """
    comp = ccd.get(res_name)
    if comp is None:
        return None
    return comp.type