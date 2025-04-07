"""Sterics module for structural clash detection and validation."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dnaflex.structure.structure import Structure

def find_clashing_chains(structure: Structure, cutoff: float = 1.5) -> List[str]:
    """Find chains with severe clashes.
    
    Args:
        structure: Structure object to analyze
        cutoff: Distance cutoff in Å for clash detection (default 1.5Å)
        
    Returns:
        List of chain IDs with severe clashes
    """
    clashing_chains = set()
    
    # Get pairwise distances between atoms of different chains
    coords = structure.get_atom_coordinates()
    chain_ids = structure.get_chain_ids()
    
    for i, chain_i in enumerate(chain_ids):
        for j in range(i + 1, len(chain_ids)):
            chain_j = chain_ids[j]
            
            # Get atom coordinates for each chain
            atoms_i = coords[structure.chain_id == chain_i]
            atoms_j = coords[structure.chain_id == chain_j]
            
            # Calculate pairwise distances
            if len(atoms_i) > 0 and len(atoms_j) > 0:
                clashes = _detect_clashes(atoms_i, atoms_j, cutoff)
                
                if clashes:
                    clashing_chains.add(chain_i)
                    clashing_chains.add(chain_j)
                    
    return list(clashing_chains)

def _detect_clashes(coords1: np.ndarray, coords2: np.ndarray, 
                   cutoff: float) -> List[Tuple[int, int]]:
    """Detect clashes between two sets of coordinates.
    
    Args:
        coords1: First set of coordinates [N, 3]
        coords2: Second set of coordinates [M, 3]
        cutoff: Distance cutoff for clash detection
        
    Returns:
        List of (i, j) indices where atoms clash
    """
    clashes = []
    
    # Efficient distance calculation using broadcasting
    diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Find pairs closer than cutoff
    close_pairs = np.where(distances < cutoff)
    
    for i, j in zip(*close_pairs):
        clashes.append((i, j))
        
    return clashes

def calculate_steric_energy(structure: Structure) -> Dict[str, float]:
    """Calculate steric energy of structure.
    
    Args:
        structure: Structure to analyze
        
    Returns:
        Dictionary with energy components
    """
    # Get atomic coordinates and types
    coords = structure.get_atom_coordinates()
    atom_types = structure.get_atom_types()
    
    # Calculate pairwise interactions
    energy_components = {
        'vdw': _calculate_vdw_energy(coords, atom_types),
        'electrostatic': _calculate_electrostatic_energy(coords, atom_types, structure),
        'total': 0.0
    }
    
    energy_components['total'] = energy_components['vdw'] + energy_components['electrostatic']
    
    return energy_components

def _calculate_vdw_energy(coords: np.ndarray, 
                         atom_types: List[str]) -> float:
    """Calculate van der Waals energy."""
    # Lennard-Jones parameters for common atom types
    lj_params = {
        'C': {'sigma': 1.7, 'epsilon': 0.1094},
        'N': {'sigma': 1.55, 'epsilon': 0.1700},
        'O': {'sigma': 1.52, 'epsilon': 0.2100},
        'P': {'sigma': 1.80, 'epsilon': 0.2000},
        'S': {'sigma': 1.80, 'epsilon': 0.2500},
        'H': {'sigma': 1.20, 'epsilon': 0.0157},
    }
    
    energy = 0.0
    
    # Calculate pairwise interactions
    for i in range(len(coords)):
        type_i = atom_types[i][0]  # Take first letter as element
        if type_i not in lj_params:
            continue
            
        for j in range(i + 1, len(coords)):
            type_j = atom_types[j][0]
            if type_j not in lj_params:
                continue
                
            # Combine LJ parameters
            sigma = (lj_params[type_i]['sigma'] + lj_params[type_j]['sigma']) / 2
            epsilon = np.sqrt(lj_params[type_i]['epsilon'] * lj_params[type_j]['epsilon'])
            
            # Calculate distance
            r = np.linalg.norm(coords[i] - coords[j])
            
            if r > 0:
                # Calculate LJ potential
                sr6 = (sigma/r)**6
                energy += 4 * epsilon * (sr6**2 - sr6)
                
    return energy

def _calculate_electrostatic_energy(coords: np.ndarray, 
                                  atom_types: List[str],
                                  structure: Structure) -> float:
    """Calculate electrostatic energy using Coulomb's law."""
    # Partial charges for common atom types
    partial_charges = {
        'N': -0.5, 'O': -0.5, 'P': 1.0,
        'C': 0.0, 'H': 0.1, 'S': 0.0
    }
    
    # Dielectric constant (distance-dependent)
    eps0 = 8.854e-12  # Vacuum permittivity
    k = 8.988e9  # Coulomb's constant
    
    energy = 0.0
    
    for i in range(len(coords)):
        type_i = atom_types[i][0]
        if type_i not in partial_charges:
            continue
            
        for j in range(i + 1, len(coords)):
            type_j = atom_types[j][0]
            if type_j not in partial_charges:
                continue
                
            # Get charges
            q1 = partial_charges[type_i]
            q2 = partial_charges[type_j]
            
            # Calculate distance
            r = np.linalg.norm(coords[i] - coords[j])
            
            if r > 0:
                # Distance-dependent dielectric
                eps_r = 4 * r  # Simple distance-dependent model
                
                # Calculate Coulomb energy
                energy += k * (q1 * q2) / (eps_r * r)
                
    return energy

def optimize_sterics(structure: Structure, 
                    max_iter: int = 100) -> Structure:
    """Optimize structure to minimize steric clashes.
    
    Args:
        structure: Structure to optimize
        max_iter: Maximum number of optimization iterations
        
    Returns:
        Optimized structure
    """
    coords = structure.get_atom_coordinates()
    atom_types = structure.get_atom_types()
    
    # Simple gradient descent optimization
    learning_rate = 0.01
    prev_energy = float('inf')
    
    for _ in range(max_iter):
        # Calculate current energy and gradients
        energy = calculate_steric_energy(structure)['total']
        
        if energy > prev_energy:
            learning_rate *= 0.5
            
        if abs(energy - prev_energy) < 1e-6:
            break
            
        # Calculate numerical gradients
        gradients = _calculate_gradients(coords, atom_types)
        
        # Update coordinates
        coords = coords - learning_rate * gradients
        
        # Update structure
        structure = structure.with_coordinates(coords)
        prev_energy = energy
        
    return structure

def _calculate_gradients(coords: np.ndarray, 
                        atom_types: List[str],
                        delta: float = 1e-5) -> np.ndarray:
    """Calculate numerical gradients for optimization."""
    gradients = np.zeros_like(coords)
    
    for i in range(len(coords)):
        for j in range(3):  # x, y, z coordinates
            # Forward difference
            coords[i, j] += delta
            energy_plus = calculate_steric_energy(Structure(coords, atom_types))['total']
            
            # Backward difference
            coords[i, j] -= 2*delta
            energy_minus = calculate_steric_energy(Structure(coords, atom_types))['total']
            
            # Reset coordinates
            coords[i, j] += delta
            
            # Calculate gradient
            gradients[i, j] = (energy_plus - energy_minus) / (2*delta)
            
    return gradients