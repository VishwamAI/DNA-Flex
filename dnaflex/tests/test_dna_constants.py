import pytest
import numpy as np
from dnaflex.constants import (
    atom_layouts,
    atom_types,
    chemical_component_sets,
    residue_names,
    mmcif_names,
)

def test_dna_residue_names():
    """Test DNA residue name constants."""
    # Test standard DNA residues
    assert chemical_component_sets.STANDARD_DNA == {'DA', 'DT', 'DG', 'DC'}
    
    # Test nucleotide mappings
    assert residue_names.NUCLEOTIDES['DA'] == 'A'
    assert residue_names.NUCLEOTIDES['DT'] == 'T'
    assert residue_names.NUCLEOTIDES['DG'] == 'G'
    assert residue_names.NUCLEOTIDES['DC'] == 'C'

def test_dna_atom_layouts():
    """Test DNA atom layout constants."""
    # Test DNA backbone atoms
    expected_backbone = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
    assert set(atom_layouts.DNA_BACKBONE_ATOMS) == expected_backbone
    
    # Test DNA sugar atoms
    expected_sugar = {"C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
    assert set(atom_layouts.DNA_SUGAR_ATOMS) == expected_sugar
    
    # Test DNA base atoms
    assert len(atom_layouts.DNA_BASE_ATOMS['A']) == 10  # Adenine
    assert len(atom_layouts.DNA_BASE_ATOMS['T']) == 9   # Thymine
    assert len(atom_layouts.DNA_BASE_ATOMS['G']) == 11  # Guanine
    assert len(atom_layouts.DNA_BASE_ATOMS['C']) == 8   # Cytosine

def test_dna_atom_types():
    """Test DNA atom type classifications."""
    # Test base atom classifications
    base_atoms = atom_types.get_atom_type('N1', 'DA', 'N')
    assert atom_types.ATOM_TYPES['BASE'] in base_atoms
    
    # Test backbone atom classifications
    backbone_atoms = atom_types.get_atom_type("O3'", 'DA', 'O')
    assert atom_types.ATOM_TYPES['BACKBONE'] in backbone_atoms
    assert atom_types.ATOM_TYPES['ACCEPTOR'] in backbone_atoms
    
    # Test sugar atom classifications
    sugar_atoms = atom_types.get_atom_type("C1'", 'DA', 'C')
    assert atom_types.ATOM_TYPES['SUGAR'] in sugar_atoms

def test_dna_chemical_components():
    """Test DNA chemical component classifications."""
    # Test that DNA residues are properly categorized
    assert 'DA' in chemical_component_sets.STANDARD_DNA
    assert 'DT' in chemical_component_sets.STANDARD_DNA
    assert 'DG' in chemical_component_sets.STANDARD_DNA
    assert 'DC' in chemical_component_sets.STANDARD_DNA
    
    # Test chain type
    assert mmcif_names.DNA_CHAIN == 'polydeoxyribonucleotide'
    assert mmcif_names.DNA_CHAIN in mmcif_names.NUCLEIC_ACID_CHAIN_TYPES

def test_dna_mmcif_conversion():
    """Test DNA MMCif name conversions."""
    # Test standard DNA residue conversion
    assert mmcif_names.fix_non_standard_polymer_res('DA', mmcif_names.DNA_CHAIN) == 'DA'
    
    # Test modified DNA base conversion (if implemented)
    # Add more tests for modified bases when implemented
    
def test_dna_hydrogen_bonds():
    """Test DNA base pair hydrogen bond detection."""
    # Setup example atoms for base pairing
    atoms = ['N1', 'N3', "O4'"]
    res_names = ['DA', 'DT', 'DA']
    elements = ['N', 'N', 'O']
    coords = np.array([[0,0,0], [0,0,3], [0,0,6]])  # 3Å spacing
    
    # Get H-bond pairs
    pairs = atom_types.get_hydrogen_bond_pairs(atoms, res_names, elements, coords)
    
    # Should detect N1-N3 pair but not O4' which is sugar
    assert len(pairs) > 0
    assert (0,1) in pairs  # N1-N3 pair

def test_dna_structure_validation():
    """Test DNA structure validation."""
    from dnaflex.structure.structure import DnaResidue
    
    # Test valid base type conversion
    residue = DnaResidue(
        name='DA',
        number=1,
        atoms={},
        chain_id='A'
    )
    assert residue.base_type == 'A'
    
    # Test unknown base type
    residue = DnaResidue(
        name='DX',  # Invalid base
        number=1, 
        atoms={},
        chain_id='A'
    )
    assert residue.base_type == 'N'  # Should return N for unknown