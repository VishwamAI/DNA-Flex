# DNA-Flex

A comprehensive Python library for analyzing DNA and protein flexibility, structural dynamics, and molecular interactions using deep learning and molecular dynamics approaches.

## Overview

DNA-Flex is a powerful toolkit that combines machine learning, molecular dynamics, and bioinformatics to analyze and predict:

- DNA/RNA flexibility and structural dynamics
- Protein structural analysis and dynamics
- Molecular interactions and binding predictions
- Mutation impact analysis
- Sequence-structure-function relationships

## Key Features

- **Atom Layout System**
  - Flexible atom representation for molecules
  - Support for standard and modified residues
  - Efficient handling of protein and nucleic acid structures
  
- **Bio-Language Model**
  - DNA/Protein sequence analysis
  - Embedding generation for sequences 
  - Pattern matching and motif detection
  - Structure prediction capabilities

- **Chemical Components**
  - Standard residue definitions
  - Modified amino acid handling
  - Support for ligands and non-standard components
  - Glycan component analysis

- **Structure Analysis**
  - Atomic structure manipulation
  - Secondary structure prediction
  - Domain prediction
  - Flexibility analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic DNA Analysis
```python
from dnaflex.models.protein_llm import BioLLM
from dnaflex.flexibility import analyze_dna_flexibility

# Initialize model
model = BioLLM(model_type="dna")

# Analyze DNA sequence
sequence = "ATGCTAGCTAGCT"
result = model.analyze_dna(sequence)

# Get flexibility predictions
flexibility = analyze_dna_flexibility(sequence)
```

### Protein Structure Analysis
```python
from dnaflex.models.atom_layout import AtomLayout
from dnaflex.structure import Structure

# Load and analyze protein structure
structure = Structure.from_pdb("example.pdb")
atom_layout = AtomLayout.from_structure(structure)

# Analyze protein properties
properties = model.analyze_protein(sequence)
```

## Project Structure

```
dnaflex/
├── constants/            # Constant definitions and configurations
├── data/                # Data handling and management
├── models/              # Core ML models and analysis
├── structure/           # Structure manipulation utilities
├── parsers/            # File format parsers
└── tests/              # Test suites
```

## Key Components

### Atom Layout System
The atom layout system provides a flexible framework for representing molecular structures:

- Supports various atom types and residues
- Handles protein and nucleic acid structures
- Provides efficient conversion between formats

### BioLLM Model
The biological language model offers:

- Sequence analysis capabilities
- Structure prediction
- Pattern matching
- Motif detection

### Chemical Components
Comprehensive support for:

- Standard amino acids and nucleotides
- Modified residues
- Ligands and cofactors
- Glycan components

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use DNA-Flex in your research, please cite:

```
@software{dna_flex2025,
  author = {kasinadhsarma},
  title = {DNA-Flex: A Python Library for DNA and Protein Flexibility Analysis},
  year = {2025},
  url = {https://github.com/vishwamai/DNA-Flex}
}
```
