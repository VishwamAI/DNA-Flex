"""Constants for mmCIF names and naming conventions."""

# Chain types
PROTEIN_CHAIN = 'polypeptide(L)'
RNA_CHAIN = 'polyribonucleotide'
DNA_CHAIN = 'polydeoxyribonucleotide'

# Combined sets
POLYMER_CHAIN_TYPES = {PROTEIN_CHAIN, RNA_CHAIN, DNA_CHAIN}
NUCLEIC_ACID_CHAIN_TYPES = {RNA_CHAIN, DNA_CHAIN}
LIGAND_CHAIN_TYPES = {'non-polymer', 'other'}

# Modified DNA base conversions
DNA_BASE_CONVERSIONS = {
    '5MC': 'DC',  # 5-methylcytosine to cytosine
    'M2G': 'DG',  # N2-methylguanine to guanine
    'DHU': 'DT',  # Dihydrouracil to thymine
    'CBR': 'DC',  # 5-bromocytosine to cytosine
    'CFL': 'DC',  # 5-fluorocytosine to cytosine
    '8OG': 'DG',  # 8-oxoguanine to guanine
    '6OG': 'DG',  # 6-O-methylguanine to guanine
    'PSU': 'DT',  # Pseudouridine to thymine
}

def is_standard_polymer_type(chain_type: str) -> bool:
    """Check if a chain type is a standard polymer type."""
    return chain_type in POLYMER_CHAIN_TYPES

def is_nucleic_acid_type(chain_type: str) -> bool:
    """Check if a chain type is a nucleic acid."""
    return chain_type in NUCLEIC_ACID_CHAIN_TYPES

def is_dna_base(res_name: str) -> bool:
    """Check if residue name is a standard or modified DNA base."""
    return res_name in {'DA', 'DT', 'DG', 'DC'} or res_name in DNA_BASE_CONVERSIONS

def fix_non_standard_polymer_res(res_name: str, chain_type: str) -> str:
    """Convert non-standard residue names to standard names where possible."""
    if chain_type == PROTEIN_CHAIN:
        # Handle common modified amino acids
        standard_conversions = {
            'MSE': 'MET',  # Selenomethionine to methionine
            'HYP': 'PRO',  # Hydroxyproline to proline
            'TPO': 'THR',  # Phosphothreonine to threonine
            'SEP': 'SER',  # Phosphoserine to serine
            'PTR': 'TYR',  # Phosphotyrosine to tyrosine
            'CSO': 'CYS',  # Oxidized cysteine to cysteine
            'LLP': 'LYS',  # Lysine-pyridoxal-phosphate to lysine
            'M3L': 'LYS',  # Trimethyllysine to lysine
        }
        return standard_conversions.get(res_name, res_name)
        
    elif chain_type == DNA_CHAIN:
        # Handle modified DNA bases
        return DNA_BASE_CONVERSIONS.get(res_name, res_name)
        
    elif chain_type == RNA_CHAIN:
        # Handle modified RNA bases
        standard_conversions = {
            '1MA': 'A',    # 1-methyladenosine to adenosine
            '5MC': 'C',    # 5-methylcytosine to cytosine
            'PSU': 'U',    # Pseudouridine to uridine
            '7MG': 'G',    # 7-methylguanosine to guanosine
            'H2U': 'U',    # Dihydrouridine to uridine
        }
        return standard_conversions.get(res_name, res_name)
        
    return res_name