"""Constants for chemical component sets used in molecular analysis."""

# Standard amino acids
STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

# Standard DNA nucleotides
STANDARD_DNA = {'DA', 'DT', 'DG', 'DC'}

# Standard RNA nucleotides 
STANDARD_RNA = {'A', 'U', 'G', 'C'}

# Modified amino acids commonly found in proteins
MODIFIED_AA = {
    'MSE',  # Selenomethionine
    'HYP',  # Hydroxyproline
    'PTR',  # Phosphotyrosine
    'CSO',  # Oxidized cysteine
    'TPO',  # Phosphothreonine
    'SEP',  # Phosphoserine
    'KCX',  # Lysine NZ-carboxylic acid
    'LLP',  # Lysine-pyridoxal-phosphate
}

# Common post-translational modifications
PTM_RESIDUES = {
    'PTR',  # Phosphotyrosine
    'TPO',  # Phosphothreonine 
    'SEP',  # Phosphoserine
    'NEP',  # N-linked glycosylation
    'MLY',  # Methylated lysine
    'CSW',  # Cysteine persulfide
    'M3L',  # Trimethyllysine
    'OCS',  # Cysteine sulfonic acid
}

# Common metal ions
METAL_IONS = {
    'MG',   # Magnesium
    'CA',   # Calcium
    'ZN',   # Zinc
    'FE',   # Iron
    'CU',   # Copper
    'MN',   # Manganese
    'NA',   # Sodium
    'K',    # Potassium
}

# Common cofactors
COFACTORS = {
    'ATP',  # Adenosine triphosphate
    'NAD',  # Nicotinamide adenine dinucleotide
    'FAD',  # Flavin adenine dinucleotide
    'HEM',  # Heme
    'SF4',  # Iron-sulfur cluster
    'GDP',  # Guanosine diphosphate
    'ADP',  # Adenosine diphosphate
    'COA',  # Coenzyme A
}

# Common ligands
COMMON_LIGANDS = {
    'GOL',  # Glycerol
    'SO4',  # Sulfate
    'PO4',  # Phosphate
    'EDO',  # 1,2-ethanediol
    'NAG',  # N-acetyl-D-glucosamine
    'MPD',  # 2-methyl-2,4-pentanediol
}

# Glycan components
GLYCAN_LINKING_LIGANDS = {
    'NAG',  # N-acetyl-D-glucosamine
    'MAN',  # Mannose
    'BMA',  # Beta-mannose
    'GAL',  # Galactose
    'FUC',  # Fucose
    'SIA',  # Sialic acid
}

# Other glycan-related ligands
GLYCAN_OTHER_LIGANDS = {
    'NDG',  # 2-N-acetyl-beta-D-glucosamine
    'BGC',  # Beta-D-glucose
    'GLC',  # Alpha-D-glucose
    'MAL',  # Maltose
    'SUC',  # Sucrose
}

# Common water names
WATER_NAMES = {'HOH', 'WAT', 'H2O', 'DOD'}