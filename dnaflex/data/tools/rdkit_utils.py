"""RDKit utility functions for molecular structure handling."""

from typing import Dict, List, Optional, Tuple
from rdkit import Chem

def assign_atom_names_from_graph(mol: Chem.Mol) -> Chem.Mol:
    """Assign systematic atom names to a molecule based on its graph structure.
    
    Args:
        mol: Input RDKit molecule
        
    Returns:
        Molecule with assigned atom names as properties
    """
    # First convert molecule to reference format
    mol = Chem.AddHs(mol)  # Add hydrogens
    Chem.SanitizeMol(mol)  # Clean up molecule
    
    # Generate unique names for each atom based on connectivity
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        atom_idx = atom.GetIdx()
        
        # Get list of neighbors
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        neighbors.sort()
        
        # Create name based on element and connectivity
        if atom_symbol == 'H':
            # Special case for hydrogens - name based on what they're bonded to
            parent = atom.GetNeighbors()[0]  # Hydrogens only have one neighbor
            parent_name = f"H{parent.GetIdx()+1}"
            atom.SetProp('atom_name', parent_name)
        else:
            # For non-hydrogens, name based on element and index
            atom_name = f"{atom_symbol}{atom_idx+1}"
            atom.SetProp('atom_name', atom_name)
            
    return mol

def get_element_symbol(atomic_num: int) -> str:
    """Get element symbol from atomic number.
    
    Args:
        atomic_num: Atomic number
        
    Returns:
        Element symbol
    """
    elements = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 15: 'P', 16: 'S',
        9: 'F', 17: 'Cl', 35: 'Br', 53: 'I',
        12: 'Mg', 20: 'Ca', 26: 'Fe', 30: 'Zn',
        11: 'Na', 19: 'K'
    }
    return elements.get(atomic_num, 'X')

def mol_to_atom_info(mol: Chem.Mol) -> Tuple[List[str], List[str]]:
    """Extract atom names and elements from molecule.
    
    Args:
        mol: Input RDKit molecule
        
    Returns:
        Tuple of (atom_names, atom_elements)
    """
    atom_names = []
    atom_elements = []
    
    for atom in mol.GetAtoms():
        # Get atom name from property or generate one
        if atom.HasProp('atom_name'):
            atom_names.append(atom.GetProp('atom_name'))
        else:
            atom_names.append(f"{atom.GetSymbol()}{atom.GetIdx()+1}")
            
        # Get element symbol
        atom_elements.append(get_element_symbol(atom.GetAtomicNum()))
        
    return atom_names, atom_elements

def assign_bond_orders(mol: Chem.Mol) -> None:
    """Assign bond orders to a molecule based on atom valences.
    
    Args:
        mol: RDKit molecule to modify
    """
    for atom in mol.GetAtoms():
        # Skip hydrogens
        if atom.GetAtomicNum() == 1:
            continue
            
        # Get current valence
        valence = atom.GetTotalValence()
        
        # Get expected valence based on element
        expected_valence = {
            6: 4,   # Carbon
            7: 3,   # Nitrogen  
            8: 2,   # Oxygen
            15: 5,  # Phosphorus
            16: 2,  # Sulfur
        }.get(atom.GetAtomicNum(), 0)
        
        # Add double bonds to satisfy valence
        if valence < expected_valence:
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetAtomicNum() != 1:  # Skip hydrogens
                    bond.SetBondType(Chem.BondType.DOUBLE)
                    break

def smiles_to_mol(smiles: str, sanitize: bool = True) -> Optional[Chem.Mol]:
    """Convert SMILES string to RDKit molecule.
    
    Args:
        smiles: SMILES string
        sanitize: Whether to sanitize the molecule
        
    Returns:
        RDKit molecule or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None:
            return None
            
        # Add hydrogens and assign 3D coordinates
        mol = Chem.AddHs(mol)
        
        return mol
    except:
        return None

def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> str:
    """Convert RDKit molecule to SMILES string.
    
    Args:
        mol: Input RDKit molecule
        canonical: Whether to return canonical SMILES
        
    Returns:
        SMILES string
    """
    return Chem.MolToSmiles(mol, canonical=canonical)

def get_atom_coordinates(mol: Chem.Mol) -> List[Tuple[float, float, float]]:
    """Get 3D coordinates for all atoms in a molecule.
    
    Args:
        mol: Input RDKit molecule with 3D coordinates
        
    Returns:
        List of (x,y,z) coordinate tuples for each atom
    """
    if not mol.GetNumConformers():
        return []
        
    conf = mol.GetConformer()
    return [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

def get_atom_types(mol: Chem.Mol) -> Dict[int, str]:
    """Get atom types for all atoms in a molecule.
    
    Args:
        mol: Input RDKit molecule
        
    Returns:
        Dictionary mapping atom indices to atom type strings
    """
    atom_types = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        radical = atom.GetNumRadicalElectrons()
        
        atom_type = symbol
        if charge > 0:
            atom_type += f"+{charge}"
        elif charge < 0:
            atom_type += str(charge)
        if radical > 0:
            atom_type += f".{radical}"
            
        atom_types[idx] = atom_type
        
    return atom_types