    # DNA-Flex API Reference

## Core Components

### Flexibility Analysis

#### FlexibilityAnalyzer
Main class for analyzing DNA flexibility.

```python
from dnaflex.flexibility import FlexibilityAnalyzer

analyzer = FlexibilityAnalyzer(structure)
```

Methods:
- `predict_flexibility(chain)`: Calculate flexibility scores for each base
- `calculate_base_step_parameters(chain)`: Calculate base step parameters
- `identify_flexible_regions(chain, window_size=4, threshold=0.6)`: Find flexible regions

### Structure Handling

#### DnaStructure
Container for DNA molecule structure.

```python
from dnaflex.structure.structure import DnaStructure

structure = DnaStructure()
```

Methods:
- `add_chain(chain)`: Add a new chain to structure
- `get_chain(chain_id)`: Get chain by ID
- `calculate_center_of_mass()`: Calculate structure's center of mass
- `calculate_radius_of_gyration()`: Calculate radius of gyration

#### DnaChain
Represents a DNA chain (strand).

Methods:
- `add_residue(residue)`: Add a residue to chain
- `get_residue(number)`: Get residue by number
- `sequence`: Property returning DNA sequence

### Data Management

#### DataManager
Central manager for data operations.

```python
from dnaflex.data import DataManager

manager = DataManager(data_dir="path/to/data", email="user@example.com")
```

Methods:
- `get_sequence(identifier, source)`: Get sequence data
- `get_structure(identifier, source)`: Get structure data
- `save_sequence(sequence_data, identifier)`: Save sequence
- `save_structure(structure_data, identifier)`: Save structure

#### DataCache
Cache handler for data operations.

Methods:
- `get(key)`: Retrieve cached item
- `set(key, value)`: Store item in cache
- `invalidate(key)`: Remove item from cache
- `cleanup()`: Remove expired items

### Molecular Dynamics

#### MolecularDynamics
DNA molecular dynamics simulation.

```python
from dnaflex.models.dynamics import MolecularDynamics

md = MolecularDynamics()
```

Methods:
- `simulate(sequence)`: Run molecular dynamics simulation
- `_simulate_thermal_fluctuations(sequence_length)`: Calculate thermal fluctuations
- `_calculate_energies(sequence)`: Calculate energy components

### Drug Binding Analysis

#### BindingAnalyzer
Analyze drug binding to DNA.

Methods:
- `_flexibility_score(sequence, site)`: Calculate flexibility-based binding score
- `_calculate_binding_propensity(scored_sites)`: Calculate binding propensity

### Data Processing

#### DNADataProcessor
Process DNA sequence and structure data.

```python
from dnaflex.models.dna_data_processing import DNADataProcessor

processor = DNADataProcessor(config)
```

Methods:
- `process_sequence(sequence)`: Process DNA sequence into features
- `process_structure(structure)`: Process structure data
- `_calculate_gc_content(sequence)`: Calculate GC content
- `_extract_kmers(sequence)`: Extract k-mer frequencies

## Constants and Utilities

### Atom Types and Layouts
```python
from dnaflex.constants import atom_types, atom_layouts

# Access constants
DNA_RESIDUES = atom_types.DNA_RESIDUES
BACKBONE_ATOMS = atom_layouts.BACKBONE_ATOMS
```

### Common Utilities
```python
from dnaflex.models.components import utils

# Calculate distances
distances = utils.compute_pairwise_distances(coords)

# Normalize vectors
normalized = utils.normalize_vector(vectors)
```

## Advanced Features

### Multiple Sequence Alignment
```python
from dnaflex.data.cpp.msa_profile_pybind import MSAProfile

profile = MSAProfile()
profile.add_sequence("ATCG")
profile.compute_profile()
```

Methods:
- `add_sequence(sequence)`: Add sequence to alignment
- `compute_profile()`: Compute alignment profile
- `get_conservation_scores()`: Get conservation scores
- `get_consensus_sequence()`: Get consensus sequence

## Error Handling

Most methods will raise appropriate exceptions with descriptive messages:

- `ValueError`: Invalid input values
- `FileNotFoundError`: Missing files
- `RuntimeError`: Computation errors
- `KeyError`: Missing keys in data structures

## Configuration

### BaseConfig
Base configuration class.

```python
from dnaflex.common.base_config import BaseConfig

class MyConfig(BaseConfig):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
```

Methods:
- `create(**kwargs)`: Create config with defaults
- `to_dict()`: Convert to dictionary
- `from_dict(config_dict)`: Create from dictionary
- `update(**kwargs)`: Update config values