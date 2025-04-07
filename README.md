# DNA-Flex

DNA-Flex is a Python package for analyzing and predicting DNA flexibility and structural dynamics. It provides tools for molecular dynamics simulations, flexibility analysis, and structure-based predictions for DNA sequences.

## Features

- DNA flexibility analysis based on sequence and structure
- Molecular dynamics simulations for DNA molecules
- Multiple sequence alignment and profile analysis
- Structure-based predictions and analysis
- Drug binding site prediction
- Efficient data caching and management
- C++ accelerated computations via pybind11

## Installation

```bash
pip install dnaflex
```

### Development Installation

```bash
git clone https://github.com/yourusername/DNA-Flex.git
cd DNA-Flex
pip install -e '.[dev]'
```

## Quick Start

```python
from dnaflex.structure import DnaStructure
from dnaflex.flexibility import FlexibilityAnalyzer

# Create structure from PDB file
structure = DnaStructure.from_pdb('dna.pdb')

# Analyze flexibility
analyzer = FlexibilityAnalyzer(structure)
flexibility_scores = analyzer.predict_flexibility(structure.chains[0])
flexible_regions = analyzer.identify_flexible_regions(structure.chains[0])

print("Flexible regions:", flexible_regions)
```

## Project Structure

```
dnaflex/
├── __init__.py
├── app.py                   # Main application entry point
├── flexibility.py          # Core flexibility analysis
├── common/                 # Common utilities and base classes
│   ├── base_config.py
├── constants/             # Constant definitions
│   ├── atom_layouts.py
│   ├── atom_types.py
│   ├── chemical_components.py
│   ├── mmcif_names.py
│   ├── residue_names.py
├── data/                  # Data loading and management
│   ├── cache.py
│   ├── loader.py
│   ├── manager.py
│   ├── providers.py
│   ├── cpp/              # C++ accelerated computations
│   │   ├── msa_profile_pybind.cc
│   │   ├── msa_profile_pybind.h
├── models/               # Analysis and prediction models
│   ├── analysis.py
│   ├── dna_data_processing.py
│   ├── dna_llm.py
│   ├── drug_binding.py
│   ├── dynamics.py
│   ├── features.py
├── parsers/              # File format parsers
│   ├── parser.py
├── structure/            # Structure representation
│   ├── structure.py
├── tests/               # Test suite
```

## Documentation

Full documentation is available at [readthedocs](https://dnaflex.readthedocs.io/).

### Core Components

1. **Flexibility Analysis:**
   - Sequence-based flexibility prediction
   - Structure-based dynamics analysis
   - Base step parameter calculations

2. **Structure Management:**
   - PDB file parsing and writing
   - Structure manipulation and analysis
   - Coordinate system transformations

3. **Data Management:**
   - Efficient caching system
   - Data providers for sequences and structures
   - File format handling

4. **Molecular Dynamics:**
   - Basic molecular dynamics simulations
   - Energy calculations
   - Thermal fluctuation analysis

## Examples

See the `notebooks/` directory for example Jupyter notebooks.

## Development

### Setting Up Development Environment

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Build C++ extensions:
```bash
cd dnaflex/data/cpp
mkdir build && cd build
cmake ..
make
```

### Running Tests

```bash
pytest tests/
```

## Future Developments

1. **Enhanced Analysis Features:**
   - Advanced machine learning models for flexibility prediction
   - Integration with external molecular dynamics engines
   - Support for RNA flexibility analysis
   - Enhanced drug binding predictions

2. **Performance Improvements:**
   - GPU acceleration for computations
   - Distributed computing support
   - Optimized memory usage for large structures

3. **New Features:**
   - Web interface for analysis
   - Interactive visualization tools
   - Integration with popular bioinformatics pipelines
   - Support for additional file formats

4. **Documentation and Training:**
   - Interactive tutorials
   - Video demonstrations
   - Comprehensive API documentation
   - Best practices guide

## Contributing

Please see CONTRIBUTING.md for guidelines on contributing to DNA-Flex.

## Citation

If you use DNA-Flex in your research, please cite:

```bibtex
@software{dnaflex2025,
  author = {Your Name},
  title = {DNA-Flex: A Python Package for DNA Flexibility Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/DNA-Flex}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
