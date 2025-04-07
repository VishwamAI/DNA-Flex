"""Defines atom types and their classifications for molecular structure analysis."""

import numpy as np
from collections import defaultdict


# Atom type classifications
ATOM_TYPES = {
    # Main categories
    'BACKBONE': 'backbone',      # Protein or nucleic acid backbone atoms
    'SIDECHAIN': 'sidechain',    # Protein sidechain atoms
    'BASE': 'base',              # Nucleic acid base atoms
    'SUGAR': 'sugar',            # Nucleic acid sugar atoms
    'TERMINAL': 'terminal',      # Terminal atoms
    'HYDROGEN': 'hydrogen',      # Hydrogen atoms
    'METAL': 'metal',            # Metal ions
    'WATER': 'water',            # Water molecules
    'LIGAND': 'ligand',          # Small molecule ligands
    
    # Subcategories
    'POLAR': 'polar',            # Polar atoms
    'NONPOLAR': 'nonpolar',      # Nonpolar atoms
    'CHARGED': 'charged',        # Charged atoms
    'AROMATIC': 'aromatic',      # Atoms in aromatic rings
    'DONOR': 'donor',            # Hydrogen bond donors
    'ACCEPTOR': 'acceptor',      # Hydrogen bond acceptors
}

# Standard protein backbone atoms
PROTEIN_BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}

# Element-based classifications
ELEMENT_CLASSES = {
    'C': [ATOM_TYPES['NONPOLAR']],
    'N': [ATOM_TYPES['POLAR'], ATOM_TYPES['ACCEPTOR']],
    'O': [ATOM_TYPES['POLAR'], ATOM_TYPES['ACCEPTOR']],
    'S': [ATOM_TYPES['POLAR']],
    'P': [ATOM_TYPES['CHARGED']],
    'H': [ATOM_TYPES['HYDROGEN']],
    'Mg': [ATOM_TYPES['METAL']],
    'Ca': [ATOM_TYPES['METAL']],
    'Zn': [ATOM_TYPES['METAL']],
    'Fe': [ATOM_TYPES['METAL']],
    'Na': [ATOM_TYPES['METAL']],
    'K': [ATOM_TYPES['METAL']],
}

# Atom name-based classifications
ATOM_CLASSES = {
    # Protein backbone
    'N': [ATOM_TYPES['BACKBONE'], ATOM_TYPES['DONOR']],
    'CA': [ATOM_TYPES['BACKBONE']],
    'C': [ATOM_TYPES['BACKBONE']],
    'O': [ATOM_TYPES['BACKBONE'], ATOM_TYPES['ACCEPTOR']],
    
    # DNA/RNA backbone
    'P': [ATOM_TYPES['BACKBONE'], ATOM_TYPES['CHARGED']],
    'OP1': [ATOM_TYPES['BACKBONE'], ATOM_TYPES['CHARGED'], ATOM_TYPES['ACCEPTOR']],
    'OP2': [ATOM_TYPES['BACKBONE'], ATOM_TYPES['CHARGED'], ATOM_TYPES['ACCEPTOR']],
    "O5'": [ATOM_TYPES['BACKBONE'], ATOM_TYPES['ACCEPTOR']],
    "O3'": [ATOM_TYPES['BACKBONE'], ATOM_TYPES['ACCEPTOR']],
    
    # DNA sugar atoms
    "C1'": [ATOM_TYPES['SUGAR']],
    "C2'": [ATOM_TYPES['SUGAR']],
    "C3'": [ATOM_TYPES['SUGAR']],
    "C4'": [ATOM_TYPES['SUGAR']],
    "C5'": [ATOM_TYPES['SUGAR']],
    "O4'": [ATOM_TYPES['SUGAR'], ATOM_TYPES['ACCEPTOR']],
    
    # Special cases
    'OXT': [ATOM_TYPES['TERMINAL'], ATOM_TYPES['CHARGED'], ATOM_TYPES['ACCEPTOR']],
    'H': [ATOM_TYPES['BACKBONE'], ATOM_TYPES['HYDROGEN'], ATOM_TYPES['DONOR']],
}

# Aromatic atoms in protein sidechains
AROMATIC_ATOMS = [
    # Phenylalanine
    'PHE-CG', 'PHE-CD1', 'PHE-CD2', 'PHE-CE1', 'PHE-CE2', 'PHE-CZ',
    # Tyrosine
    'TYR-CG', 'TYR-CD1', 'TYR-CD2', 'TYR-CE1', 'TYR-CE2', 'TYR-CZ',
    # Tryptophan
    'TRP-CG', 'TRP-CD1', 'TRP-CD2', 'TRP-CE2', 'TRP-CE3', 'TRP-CZ2', 'TRP-CZ3', 'TRP-CH2',
    # Histidine
    'HIS-CG', 'HIS-CD2', 'HIS-CE1', 'HIS-ND1', 'HIS-NE2',
]

# Charged atoms
CHARGED_ATOMS = [
    # Positive
    'ARG-NH1', 'ARG-NH2', 'LYS-NZ', 'HIS-ND1', 'HIS-NE2',
    # Negative
    'ASP-OD1', 'ASP-OD2', 'GLU-OE1', 'GLU-OE2',
]

# Polar atoms in sidechains
POLAR_SIDECHAIN_ATOMS = [
    'SER-OG', 'THR-OG1', 'ASN-OD1', 'ASN-ND2', 'GLN-OE1', 'GLN-NE2',
    'TYR-OH', 'CYS-SG', 'MET-SD'
]

"""Constants for atom types and protonation states."""

# Standard protonation hydrogens per residue type
PROTONATION_HYDROGENS = {
    'ASP': {'HD2'},  # Aspartic acid
    'GLU': {'HE2'},  # Glutamic acid
    'HIS': {'HD1', 'HE2'},  # Histidine
    'LYS': {'HZ3'},  # Lysine
    'CYS': {'HG'},  # Cysteine
    'SER': {'HG'},  # Serine
    'THR': {'HG1'},  # Threonine
    'TYR': {'HH'},  # Tyrosine
}

# Standard heavy atom types per residue type
HEAVY_ATOM_TYPES = {
    'ALA': {'N', 'CA', 'C', 'O', 'CB'},
    'ARG': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'},
    'ASN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'},
    'ASP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'},
    'CYS': {'N', 'CA', 'C', 'O', 'CB', 'SG'},
    'GLN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'},
    'GLU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'},
    'GLY': {'N', 'CA', 'C', 'O'},
    'HIS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'},
    'ILE': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'},
    'LEU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'},
    'LYS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'},
    'MET': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'},
    'PHE': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
    'PRO': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'},
    'SER': {'N', 'CA', 'C', 'O', 'CB', 'OG'},
    'THR': {'N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'},
    'TRP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'},
    'TYR': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'},
    'VAL': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'},
}

# DNA base atom types with hydrogen bonding properties
DNA_BASE_ATOMS = {
    'DA': {  # Adenine
        'atoms': {'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', 'N9'},
        'donors': {'N1', 'N6'},  # N1 is donor to Thymine N3, N6 to Thymine O4
        'acceptors': {'N7'}  # N7 can accept in non-WC base pairs
    },
    'DT': {  # Thymine
        'atoms': {'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6'},
        'donors': {},  # No donors in canonical WC base pairing
        'acceptors': {'O2', 'O4', 'N3'}  # N3 accepts from Adenine N1
    },
    'DG': {  # Guanine
        'atoms': {'N1', 'C2', 'N2', 'N3', 'C4', 'C5', 'C6', 'O6', 'N7', 'C8', 'N9'},
        'donors': {'N1', 'N2'},
        'acceptors': {'O6', 'N3', 'N7'}
    },
    'DC': {  # Cytosine
        'atoms': {'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'},
        'donors': {'N4'},
        'acceptors': {'O2', 'N3'}
    }
}

# Standard backbone atoms for nucleic acids
DNA_BACKBONE_ATOMS = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}

# Atom hybridization states
HYBRIDIZATION = {
    'sp3': {'CA', 'CB', "C1'", "C2'", "C3'", "C4'", "C5'"},
    'sp2': {'C', 'CD', 'CG', 'CE', 'CZ'},
    'aromatic': {'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
}

# Valid atom elements
VALID_ELEMENTS = {
    'H', 'C', 'N', 'O', 'S', 'P', 'Fe', 'Zn', 'Cu', 'Mg', 'Ca', 'Na', 'K', 'Cl', 'F', 'Br', 'I'
}

# Maximum number of atoms per layout
MAX_PROTEIN_ATOMS = 37  # Maximum for standard amino acids including H atoms
MAX_DNA_ATOMS = 30     # Maximum for DNA nucleotides including backbone

def get_atom_type(atom_name, res_name=None, element=None):
    """Determine atom type classifications for a given atom.
    
    Args:
        atom_name: Atom name (e.g., 'CA', 'N', 'O')
        res_name: Residue name (e.g., 'ALA', 'DA')
        element: Element symbol (e.g., 'C', 'N', 'O')
    
    Returns:
        List of atom type classifications
    """
    classifications = []
    
    # Add atom-specific classifications
    if atom_name in ATOM_CLASSES:
        classifications.extend(ATOM_CLASSES[atom_name])
    
    # Add element-specific classifications
    if element and element in ELEMENT_CLASSES:
        for class_type in ELEMENT_CLASSES[element]:
            if class_type not in classifications:
                classifications.append(class_type)
    
    # Handle DNA base atoms and hydrogen bonding
    if res_name in DNA_BASE_ATOMS:
        base_info = DNA_BASE_ATOMS[res_name]
        if atom_name in base_info['atoms']:
            if ATOM_TYPES['BASE'] not in classifications:
                classifications.append(ATOM_TYPES['BASE'])
            # Add donor/acceptor classifications with priority over element-based ones
            if ATOM_TYPES['DONOR'] in classifications:
                classifications.remove(ATOM_TYPES['DONOR'])
            if ATOM_TYPES['ACCEPTOR'] in classifications:
                classifications.remove(ATOM_TYPES['ACCEPTOR'])
            if atom_name in base_info['donors']:
                classifications.append(ATOM_TYPES['DONOR'])
            if atom_name in base_info['acceptors']:
                classifications.append(ATOM_TYPES['ACCEPTOR'])
    
    # Handle protein sidechain atoms
    if res_name in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL']:
        if atom_name not in PROTEIN_BACKBONE_ATOMS:
            if ATOM_TYPES['SIDECHAIN'] not in classifications:
                classifications.append(ATOM_TYPES['SIDECHAIN'])
    
    # Check for aromatic atoms
    res_atom = f"{res_name}-{atom_name}" if res_name else None
    if res_atom and res_atom in AROMATIC_ATOMS:
        if ATOM_TYPES['AROMATIC'] not in classifications:
            classifications.append(ATOM_TYPES['AROMATIC'])
    
    # Check for charged atoms
    if res_atom and res_atom in CHARGED_ATOMS:
        if ATOM_TYPES['CHARGED'] not in classifications:
            classifications.append(ATOM_TYPES['CHARGED'])
    
    # Check for polar sidechain atoms
    if res_atom and res_atom in POLAR_SIDECHAIN_ATOMS:
        if ATOM_TYPES['POLAR'] not in classifications:
            classifications.append(ATOM_TYPES['POLAR'])
    
    # Detect hydrogens
    if atom_name.startswith('H'):
        if ATOM_TYPES['HYDROGEN'] not in classifications:
            classifications.append(ATOM_TYPES['HYDROGEN'])
        if ATOM_TYPES['DONOR'] not in classifications:
            classifications.append(ATOM_TYPES['DONOR'])
    
    return classifications

def classify_structure(atom_names, res_names=None, elements=None):
    """Classify atoms in a structure by type.
    
    Args:
        atom_names: List of atom names
        res_names: List of residue names (optional)
        elements: List of element symbols (optional)
    
    Returns:
        Dictionary mapping atom types to indices
    """
    if res_names is None:
        res_names = [None] * len(atom_names)
    if elements is None:
        elements = [None] * len(atom_names)
    
    classification = defaultdict(list)
    
    for i, (atom_name, res_name, element) in enumerate(zip(atom_names, res_names, elements)):
        atom_types = get_atom_type(atom_name, res_name, element)
        for atom_type in atom_types:
            classification[atom_type].append(i)
    
    return {k: np.array(v) for k, v in classification.items()}

def get_hydrogen_bond_pairs(atom_names, res_names=None, elements=None, coords=None, cutoff=3.5):
    """Identify potential hydrogen bond donor-acceptor pairs.
    
    Args:
        atom_names: List of atom names
        res_names: List of residue names (optional)
        elements: List of element symbols (optional)
        coords: Array of coordinates (optional)
        cutoff: Distance cutoff for hydrogen bonds (Angstroms)
    
    Returns:
        List of (donor_idx, acceptor_idx) pairs
    """
    if coords is None:
        return []  # Need coordinates to compute distances
    
    classification = classify_structure(atom_names, res_names, elements)
    
    donors = classification.get(ATOM_TYPES['DONOR'], [])
    acceptors = classification.get(ATOM_TYPES['ACCEPTOR'], [])
    
    h_bonds = []
    
    for d_idx in donors:
        for a_idx in acceptors:
            # Skip self-interactions
            if d_idx == a_idx:
                continue
                
            # Calculate distance
            dist = np.linalg.norm(coords[d_idx] - coords[a_idx])
            
            if dist <= cutoff:
                h_bonds.append((d_idx, a_idx))
    
    return h_bonds

def get_aromatic_interactions(atom_names, res_names, coords=None, cutoff=6.0):
    """Identify potential aromatic-aromatic interactions.
    
    Args:
        atom_names: List of atom names
        res_names: List of residue names
        coords: Array of coordinates (optional)
        cutoff: Distance cutoff for aromatic interactions (Angstroms)
    
    Returns:
        List of (res_i, res_j) pairs with aromatic interactions
    """
    if coords is None:
        return []  # Need coordinates to compute distances
    
    classification = classify_structure(atom_names, res_names)
    
    aromatic_atoms = classification.get(ATOM_TYPES['AROMATIC'], [])
    
    # Group aromatic atoms by residue
    aromatic_residues = {}
    for idx in aromatic_atoms:
        res_name = res_names[idx]
        res_num = idx // len(atom_names)  # Assuming sequential residue numbering
        if res_num not in aromatic_residues:
            aromatic_residues[res_num] = []
        aromatic_residues[res_num].append(idx)
    
    interactions = []
    
    # Check pairs of aromatic residues
    res_nums = list(aromatic_residues.keys())
    for i in range(len(res_nums)):
        for j in range(i+1, len(res_nums)):
            res_i = res_nums[i]
            res_j = res_nums[j]
            
            # Calculate minimum distance between any atoms in the two residues
            min_dist = float('inf')
            for atom_i in aromatic_residues[res_i]:
                for atom_j in aromatic_residues[res_j]:
                    dist = np.linalg.norm(coords[atom_i] - coords[atom_j])
                    min_dist = min(min_dist, dist)
            
            # If close enough, consider it an interaction
            if min_dist <= cutoff:
                interactions.append((res_i, res_j))
    
    return interactions