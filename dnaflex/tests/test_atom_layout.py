"""Tests for atom layout functionality."""

import pytest
import numpy as np
from dnaflex.models.atom_layout.atom_layout import AtomLayout, Residues, GatherInfo
from dnaflex.constants import atom_layouts, atom_types
from dnaflex.structure import structure

@pytest.fixture
def sample_protein_layout():
    """Create a sample protein atom layout for testing."""
    return AtomLayout(
        atom_name=np.array(['N', 'CA', 'C', 'O', 'CB'], dtype=object),
        res_id=np.array([1, 1, 1, 1, 1], dtype=int),
        chain_id=np.array(['A', 'A', 'A', 'A', 'A'], dtype=object),
        atom_element=np.array(['N', 'C', 'C', 'O', 'C'], dtype=object),
        res_name=np.array(['ALA', 'ALA', 'ALA', 'ALA', 'ALA'], dtype=object),
        chain_type=np.array(['polypeptide(L)']*5, dtype=object)
    )

@pytest.fixture
def sample_dna_layout():
    """Create a sample DNA atom layout for testing."""
    return AtomLayout(
        atom_name=np.array(["P", "O5'", "C5'", "C4'", "O4'"], dtype=object),
        res_id=np.array([1, 1, 1, 1, 1], dtype=int),
        chain_id=np.array(['B', 'B', 'B', 'B', 'B'], dtype=object),
        atom_element=np.array(['P', 'O', 'C', 'C', 'O'], dtype=object),
        res_name=np.array(['DA', 'DA', 'DA', 'DA', 'DA'], dtype=object),
        chain_type=np.array(['polydeoxyribonucleotide']*5, dtype=object)
    )

@pytest.fixture
def sample_residues():
    """Create sample residues for testing."""
    return Residues(
        res_name=np.array(['ALA', 'GLY', 'SER'], dtype=object),
        res_id=np.array([1, 2, 3], dtype=int),
        chain_id=np.array(['A', 'A', 'A'], dtype=object),
        chain_type=np.array(['polypeptide(L)']*3, dtype=object),
        is_start_terminus=np.array([True, False, False], dtype=bool),
        is_end_terminus=np.array([False, False, True], dtype=bool),
    )

def test_atom_layout_initialization(sample_protein_layout):
    """Test AtomLayout initialization and basic properties."""
    # Test shape property
    assert sample_protein_layout.shape == (5,)
    
    # Test array types
    assert sample_protein_layout.atom_name.dtype == object
    assert sample_protein_layout.res_id.dtype == int
    assert sample_protein_layout.chain_id.dtype == object

def test_atom_layout_validation():
    """Test validation of atom layout data."""
    with pytest.raises(ValueError):
        # Test mismatched shapes
        AtomLayout(
            atom_name=np.array(['N', 'CA'], dtype=object),
            res_id=np.array([1], dtype=int),  # Wrong shape
            chain_id=np.array(['A', 'A'], dtype=object)
        )

def test_atom_layout_indexing(sample_protein_layout):
    """Test indexing operations on AtomLayout."""
    # Test single index
    first_atom = sample_protein_layout[0:1]  # Get slice instead of single element
    assert isinstance(first_atom, AtomLayout)
    assert first_atom.atom_name[0] == 'N'
    assert first_atom.shape == (1,)
    
    # Test slice
    subset = sample_protein_layout[1:3]
    assert subset.shape == (2,)
    assert np.array_equal(subset.atom_name, np.array(['CA', 'C'], dtype=object))
    
    # Test boolean indexing
    mask = sample_protein_layout.atom_element == 'C'
    carbon_atoms = sample_protein_layout[mask]
    assert all(elem == 'C' for elem in carbon_atoms.atom_element)
    
    # Test advanced indexing
    indices = np.array([0, 2, 4])
    selected = sample_protein_layout[indices]
    assert selected.shape == (3,)
    assert np.array_equal(selected.atom_name, np.array(['N', 'C', 'CB'], dtype=object))

def test_atom_layout_equality(sample_protein_layout):
    """Test equality comparison between atom layouts."""
    # Same layout should be equal
    assert sample_protein_layout == sample_protein_layout
    
    # Different layout should not be equal
    different = AtomLayout(
        atom_name=np.array(['N', 'CA', 'C', 'O', 'CG'], dtype=object),
        res_id=np.array([1, 1, 1, 1, 1], dtype=int),
        chain_id=np.array(['A', 'A', 'A', 'A', 'A'], dtype=object)
    )
    assert sample_protein_layout != different

def test_residue_handling(sample_residues):
    """Test residue handling functionality."""
    # Test residue properties
    assert len(sample_residues.res_name) == 3
    assert list(sample_residues.res_name) == ['ALA', 'GLY', 'SER']
    
    # Test terminus flags
    assert sample_residues.is_start_terminus[0]  # ALA is start
    assert sample_residues.is_end_terminus[2]    # SER is end

def test_gather_info():
    """Test GatherInfo functionality."""
    gather_info = GatherInfo(
        gather_idxs=np.array([0, 1, 2]),
        gather_mask=np.array([True, True, False]),
        input_shape=np.array([3])
    )
    
    # Test properties
    assert gather_info.shape == (3,)
    assert np.all(gather_info.gather_idxs == np.array([0, 1, 2]))
    assert np.all(gather_info.gather_mask == np.array([True, True, False]))

def test_atom_layout_padding(sample_protein_layout):
    """Test padding operations on atom layouts."""
    # Pad to larger shape
    padded = sample_protein_layout.copy_and_pad_to((10,))
    assert padded.shape == (10,)
    assert np.all(padded.atom_name[5:] == '')
    
    # Test invalid padding
    with pytest.raises(ValueError):
        sample_protein_layout.copy_and_pad_to((3,))  # Can't pad to smaller shape

def test_dna_specific_features(sample_dna_layout):
    """Test DNA-specific layout features."""
    # Verify DNA backbone atoms
    backbone_atoms = set(atom_layouts.DNA_BACKBONE_ATOMS)
    layout_atoms = set(sample_dna_layout.atom_name)
    assert layout_atoms.issubset(backbone_atoms)
    
    # Test DNA chain type
    assert sample_dna_layout.chain_type[0] == 'polydeoxyribonucleotide'

def test_atom_type_validation(sample_protein_layout):
    """Test validation of atom types and elements."""
    # Verify protein backbone atoms
    backbone_atoms = atom_types.PROTEIN_BACKBONE_ATOMS
    assert set(sample_protein_layout.atom_name[:4]).issubset(backbone_atoms)
    
    # Test element types
    carbon_atoms = sample_protein_layout.atom_name[
        sample_protein_layout.atom_element == 'C'
    ]
    assert 'CA' in carbon_atoms
    assert 'CB' in carbon_atoms

def test_residue_conversions(sample_residues, sample_protein_layout):
    """Test conversions between residues and atom layouts."""
    # Get atoms for a specific residue
    res_idx = 0
    res_atoms = sample_protein_layout[
        sample_protein_layout.res_id == sample_residues.res_id[res_idx]
    ]
    assert res_atoms.res_name[0] == sample_residues.res_name[res_idx]
    
    # Verify chain consistency
    assert res_atoms.chain_id[0] == sample_residues.chain_id[res_idx]

def test_array_conversion(sample_protein_layout):
    """Test conversion to/from numpy arrays."""
    # Convert to array
    arr = sample_protein_layout.to_array()
    assert arr.shape[0] == 6  # number of fields
    assert arr.dtype == object
    
    # Convert back to AtomLayout
    reconverted = AtomLayout.from_array(arr)
    assert reconverted == sample_protein_layout

def test_structure_integration(sample_protein_layout):
    """Test integration with Structure class."""
    # Create minimal structure
    struct = structure.from_atom_arrays(
        name='test',
        all_residues={'A': [('ALA', 1)]},
        chain_id=sample_protein_layout.chain_id,
        chain_type=sample_protein_layout.chain_type,
        res_id=sample_protein_layout.res_id,
        res_name=sample_protein_layout.res_name,
        atom_name=sample_protein_layout.atom_name,
        atom_element=sample_protein_layout.atom_element,
        atom_x=np.zeros(5),
        atom_y=np.zeros(5),
        atom_z=np.zeros(5),
        atom_b_factor=np.zeros(5)
    )
    
    # Convert back to AtomLayout
    layout = AtomLayout(
        atom_name=struct.atom_name,
        res_id=struct.res_id,
        chain_id=struct.chain_id,
        atom_element=struct.atom_element,
        res_name=struct.res_name,
        chain_type=struct.chain_type
    )
    
    assert layout == sample_protein_layout