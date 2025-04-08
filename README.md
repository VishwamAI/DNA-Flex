# DNA-Flex

DNA-Flex is a powerful Python library for analyzing DNA sequence flexibility, structure prediction, and advanced genomic analysis using machine learning approaches.

## Features

- 🧬 **DNA Sequence Analysis**
  - Comprehensive sequence analysis
  - Structure prediction
  - Flexibility assessment
  - Mutation impact analysis

- 🤖 **AI-Powered Analysis**
  - Machine learning models for sequence analysis
  - Deep learning-based structure prediction
  - NLP-based sequence pattern recognition

- ⚡ **High-Performance Computing**
  - JAX-accelerated computations
  - Async background processing
  - Batch analysis support
  - C++ optimized core components

- 🔄 **API Integration**
  - RESTful API with FastAPI
  - JWT Authentication
  - Rate limiting
  - Background task processing
  - Swagger/OpenAPI documentation

## Quick Start

### Installation

```bash
# Basic installation
pip install dnaflex

# Development installation
git clone https://github.com/vishwamai/DNA-Flex.git
cd DNA-Flex
pip install -e '.[dev]'
```

### Basic Usage

```python
from dnaflex.models.analysis import analyze
from dnaflex.models.dynamics import molecular_dynamics

# Analyze DNA sequence
sequence = "ATGCTAGCTAGCT"
result = analyze(sequence)
print(f"GC Content: {result['gc_content']}%")
print(f"Flexibility Score: {result['flexibility_score']}")

# Run molecular dynamics simulation
dynamics = molecular_dynamics.simulate(sequence)
```

### API Usage

```python
import requests

# Get authentication token
response = requests.post(
    "http://localhost:8000/token",
    data={"username": "your-username", "password": "your-password"}
)
token = response.json()["access_token"]

# Analyze sequence
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "http://localhost:8000/predict",
    headers=headers,
    json={"sequence": "ATGCTAGCTAGCT"}
)
result = response.json()
```

## Architecture

DNA-Flex is built with a modular architecture:

- **Core Analysis Engine**: Built in Python with C++ extensions
- **Machine Learning Models**: Using JAX and deep learning
- **REST API**: FastAPI with async processing
- **Data Management**: Efficient caching and data providers

## API Documentation

Visit our [API Documentation](docs/api.md) for detailed endpoint information.

## Development Guide

See our [Developer Guide](docs/developer_guide.md) for setup and contribution guidelines.

## Project Structure

```
dnaflex/
├── models/          # Core analysis models
├── structure/       # Structure handling
├── flexibility/     # Flexibility analysis
├── parsers/         # File format parsers
├── data/           # Data management
└── tests/          # Test suite
```

## Configuration

Environment variables:
- `DNA_FLEX_SECRET_KEY`: JWT secret key
- `DNA_FLEX_API_PREFIX`: API prefix (default: /api/v1)
- `DNA_FLEX_RATE_LIMIT`: Rate limit (requests/minute)

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest dnaflex/tests/test_integration.py
```

## Contributing

Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DNA-Flex in your research, please cite:

```bibtex
@software{dna_flex_2025,
  title = {DNA-Flex: DNA Sequence Analysis and Structure Prediction},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/DNA-Flex}
}
```
