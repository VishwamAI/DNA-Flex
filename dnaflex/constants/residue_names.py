"""Constants for residue names and types."""

# Standard residue codes
AMINO_ACIDS = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

NUCLEOTIDES = {
    # DNA
    'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C',
    # RNA
    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C'
}

# Special residues
UNKNOWN_TYPES = {'UNK', 'UNL', 'UNX'}
WATER_TYPES = {'HOH', 'WAT', 'H2O', 'DOD'}

# Modified residues
MODIFIED_RESIDUES = {
    'MSE': 'MET',  # Selenomethionine
    'CSE': 'CYS',  # Selenocysteine
    'HYP': 'PRO',  # Hydroxyproline
    'TPO': 'THR',  # Phosphothreonine
    'SEP': 'SER',  # Phosphoserine
    'PTR': 'TYR',  # Phosphotyrosine
    'KCX': 'LYS',  # Lysine NZ-carboxylic acid
    'LLP': 'LYS',  # Lysine-pyridoxal-phosphate
    'MLY': 'LYS',  # N-dimethyl-lysine
    'M3L': 'LYS',  # N-trimethyl-lysine
    'CSD': 'CYS',  # S-cysteinesulfinic acid
    'CSO': 'CYS',  # S-hydroxycysteine
    'OCS': 'CYS',  # Cysteine sulfonic acid
    'CSW': 'CYS',  # Cysteine-S-dioxide
}

# Mapping between one and three letter codes
ONE_TO_THREE = {one: three for three, one in AMINO_ACIDS.items()}
THREE_TO_ONE = {three: one for three, one in ONE_TO_THREE.items()}

def is_standard_aa(res_name: str) -> bool:
    """Check if residue name is a standard amino acid."""
    return res_name in AMINO_ACIDS

def is_modified_aa(res_name: str) -> bool:
    """Check if residue name is a modified amino acid."""
    return res_name in MODIFIED_RESIDUES

def is_nucleotide(res_name: str) -> bool:
    """Check if residue name is a nucleotide."""
    return res_name in NUCLEOTIDES

def is_water(res_name: str) -> bool:
    """Check if residue name represents water."""
    return res_name in WATER_TYPES

def is_unknown(res_name: str) -> bool:
    """Check if residue name represents an unknown residue type."""
    return res_name in UNKNOWN_TYPES

def standardize_name(res_name: str) -> str:
    """Convert modified residue names to standard names where possible."""
    if is_modified_aa(res_name):
        return MODIFIED_RESIDUES[res_name]
    return res_name

def to_one_letter(res_name: str) -> str:
    """Convert three letter code to one letter code."""
    standardized = standardize_name(res_name)
    if standardized in AMINO_ACIDS:
        return AMINO_ACIDS[standardized]
    if standardized in NUCLEOTIDES:
        return NUCLEOTIDES[standardized]
    raise ValueError(f"Unknown residue name: {res_name}")

def to_three_letter(one_letter: str) -> str:
    """Convert one letter code to three letter code."""
    if one_letter in ONE_TO_THREE:
        return ONE_TO_THREE[one_letter]
    raise ValueError(f"Unknown one letter code: {one_letter}")