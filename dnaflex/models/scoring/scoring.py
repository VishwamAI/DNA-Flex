"""Scoring functions for evaluating molecular structures and interactions."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dnaflex.constants import atom_types
from dnaflex.structure import structure
from dnaflex.models.components import utils

class MolecularScorer:
    """Evaluates molecular structures and interactions."""
    
    def __init__(self, 
                 distance_cutoff: float = 4.5,
                 clash_distance: float = 2.0):
        """Initialize scorer with parameters.
        
        Args:
            distance_cutoff: Maximum distance for interactions in Angstroms
            clash_distance: Minimum allowed distance between atoms in Angstroms
        """
        self.distance_cutoff = distance_cutoff
        self.clash_distance = clash_distance
        
    def score_structure(self, 
                       struc: structure.Structure,
                       include_bonds: bool = True,
                       include_angles: bool = True,
                       include_contacts: bool = True) -> Dict[str, float]:
        """Score a molecular structure.
        
        Args:
            struc: Structure to evaluate
            include_bonds: Whether to score bond lengths
            include_angles: Whether to score bond angles  
            include_contacts: Whether to score non-bonded contacts
            
        Returns:
            Dictionary of score components
        """
        scores = {}
        
        if include_bonds:
            scores['bond_score'] = self._score_bonds(struc)
            
        if include_angles:
            scores['angle_score'] = self._score_angles(struc)
            
        if include_contacts:
            scores['contact_score'] = self._score_contacts(struc)
            clash_score = self._detect_clashes(struc)
            scores['clash_score'] = clash_score
            
        # Overall score is weighted sum
        scores['total_score'] = sum(scores.values())
        
        return scores
        
    def _score_bonds(self, struc: structure.Structure) -> float:
        """Score bond lengths using harmonic potential."""
        bond_score = 0.0
        for bond in struc.bonds:
            ideal_length = atom_types.get_ideal_bond_length(
                bond.atom1.element, bond.atom2.element)
            actual_length = bond.length
            bond_score += (actual_length - ideal_length)**2
        return bond_score
        
    def _score_angles(self, struc: structure.Structure) -> float:
        """Score bond angles using harmonic potential."""
        angle_score = 0.0
        for angle in struc.get_angles():
            ideal_angle = atom_types.get_ideal_angle(
                angle.atom1.element, angle.atom2.element, angle.atom3.element)
            actual_angle = angle.value
            angle_score += (actual_angle - ideal_angle)**2
        return angle_score
        
    def _score_contacts(self, struc: structure.Structure) -> float:
        """Score favorable non-bonded contacts."""
        contact_score = 0.0
        coords = struc.get_coords()
        pairwise_dists = utils.compute_pairwise_distances(coords)
        
        # Sum up contact potential for atoms within cutoff
        in_contact = (pairwise_dists < self.distance_cutoff)
        for i, j in zip(*np.where(in_contact)):
            if i >= j:
                continue
            atom1, atom2 = struc.atoms[i], struc.atoms[j]
            contact_score += atom_types.get_contact_potential(
                atom1.element, atom2.element)
                
        return contact_score
        
    def _detect_clashes(self, struc: structure.Structure) -> float:
        """Detect steric clashes between atoms."""
        clash_score = 0.0
        coords = struc.get_coords()
        pairwise_dists = utils.compute_pairwise_distances(coords)
        
        # Penalize atom pairs that are too close
        clashing = (pairwise_dists < self.clash_distance)
        clash_score = np.sum(clashing) * 100.0  # Large penalty for clashes
        
        return clash_score