"""Constants for mmCIF names and naming conventions."""

# Chain types
PROTEIN_CHAIN = 'polypeptide(L)'
RNA_CHAIN = 'polyribonucleotide'
DNA_CHAIN = 'polydeoxyribonucleotide'

# Combined sets
POLYMER_CHAIN_TYPES = {PROTEIN_CHAIN, RNA_CHAIN, DNA_CHAIN}
NUCLEIC_ACID_CHAIN_TYPES = {RNA_CHAIN, DNA_CHAIN}
LIGAND_CHAIN_TYPES = {'non-polymer', 'other'}

def is_standard_polymer_type(chain_type: str) -> bool:
    """Check if a chain type is a standard polymer type."""
    return chain_type in POLYMER_CHAIN_TYPES

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
        
    elif chain_type in NUCLEIC_ACID_CHAIN_TYPES:
        # Handle modified nucleotides
        standard_conversions = {
            '1MA': 'A',    # 1-methyladenosine to adenosine
            '5MC': 'C',    # 5-methylcytosine to cytosine
            'PSU': 'U',    # Pseudouridine to uridine
            '7MG': 'G',    # 7-methylguanosine to guanosine
            'H2U': 'U',    # Dihydrouridine to uridine
        }
        return standard_conversions.get(res_name, res_name)
        
    return res_name