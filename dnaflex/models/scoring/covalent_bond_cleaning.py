"""Module for cleaning and validating covalent bonds in molecular structures."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dnaflex.constants import atom_types, chemical_components
from dnaflex.structure import structure

class CovalentBondCleaner:
    """Cleans and validates covalent bonds in molecular structures."""
    
    def __init__(self,
                 max_bond_length: float = 2.5,
                 min_bond_length: float = 0.8):
        """Initialize bond cleaner.
        
        Args:
            max_bond_length: Maximum allowed bond length in Angstroms
            min_bond_length: Minimum allowed bond length in Angstroms
        """
        self.max_bond_length = max_bond_length
        self.min_bond_length = min_bond_length
        
    def clean_bonds(self, 
                   struc: structure.Structure,
                   ccd: chemical_components.Ccd) -> Tuple[structure.Structure, Dict[str, int]]:
        """Clean and validate bonds in a structure.
        
        Args:
            struc: Structure to clean
            ccd: Chemical component dictionary for reference
            
        Returns:
            Tuple of (cleaned structure, cleaning statistics)
        """
        stats = {
            'total_bonds': len(struc.bonds),
            'removed_bonds': 0,
            'added_bonds': 0,
            'fixed_bonds': 0
        }
        
        # Remove invalid bonds
        valid_bonds = []
        for bond in struc.bonds:
            if self._is_valid_bond(bond):
                valid_bonds.append(bond)
            else:
                stats['removed_bonds'] += 1
                
        # Add missing bonds based on chemistry
        new_bonds = self._find_missing_bonds(struc, ccd)
        stats['added_bonds'] = len(new_bonds)
        
        # Create new structure with cleaned bonds
        cleaned_struc = struc.copy()
        cleaned_struc.bonds = valid_bonds + new_bonds
        
        # Fix bond orders and types
        self._fix_bond_types(cleaned_struc, ccd)
        stats['fixed_bonds'] = len(cleaned_struc.bonds) - stats['total_bonds']
        
        return cleaned_struc, stats
        
    def _is_valid_bond(self, bond: structure.Bond) -> bool:
        """Check if a bond is physically valid."""
        # Check bond length
        if not (self.min_bond_length <= bond.length <= self.max_bond_length):
            return False
            
        # Check if bond is chemically possible between elements
        if not atom_types.can_form_bond(bond.atom1.element, bond.atom2.element):
            return False
            
        return True
        
    def _find_missing_bonds(self,
                          struc: structure.Structure,
                          ccd: chemical_components.Ccd) -> List[structure.Bond]:
        """Find bonds that should exist based on chemistry but are missing."""
        missing_bonds = []
        coords = struc.get_coords()
        
        # Build neighbor lists for efficient search
        neighbors = self._build_neighbor_lists(coords)
        
        # Check each atom pair within bonding distance
        for i, atom1 in enumerate(struc.atoms):
            for j in neighbors[i]:
                if i >= j:
                    continue
                    
                atom2 = struc.atoms[j]
                if self._should_form_bond(atom1, atom2, ccd):
                    missing_bonds.append(structure.Bond(atom1, atom2))
                    
        return missing_bonds
        
    def _build_neighbor_lists(self, coords: np.ndarray) -> List[Set[int]]:
        """Build lists of neighboring atoms within bonding distance."""
        n_atoms = len(coords)
        neighbors = [set() for _ in range(n_atoms)]
        
        for i in range(n_atoms):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            nearby = np.where(dists < self.max_bond_length)[0]
            neighbors[i].update(nearby)
            
        return neighbors
        
    def _should_form_bond(self,
                         atom1: structure.Atom,
                         atom2: structure.Atom,
                         ccd: chemical_components.Ccd) -> bool:
        """Determine if two atoms should form a covalent bond."""
        # Check if elements can bond
        if not atom_types.can_form_bond(atom1.element, atom2.element):
            return False
            
        # Check component-specific bonding rules
        comp1 = ccd.get_component(atom1.residue.name)
        comp2 = ccd.get_component(atom2.residue.name)
        if comp1 and comp2:
            if not (comp1.allows_bond(atom1.name, atom2.name) or
                   comp2.allows_bond(atom2.name, atom1.name)):
                return False
                
        return True
        
    def _fix_bond_types(self,
                       struc: structure.Structure,
                       ccd: chemical_components.Ccd) -> None:
        """Fix bond orders and types based on chemistry."""
        for bond in struc.bonds:
            comp = ccd.get_component(bond.atom1.residue.name)
            if comp:
                bond_type = comp.get_bond_type(bond.atom1.name, bond.atom2.name)
                if bond_type:
                    bond.order = bond_type.order
                    bond.type = bond_type.type