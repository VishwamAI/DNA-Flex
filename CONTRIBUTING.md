# Contributing to DNA-Flex

Thank you for your interest in contributing to DNA-Flex! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10+
- pip
- git

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone git@github.com:your-username/DNA-Flex.git
cd DNA-Flex
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Running Tests

```bash
pytest tests/
```

### Code Style

- We follow PEP 8 guidelines
- Use type hints
- Document all functions and classes using docstrings
- Keep line length to 88 characters
- Run `black` for code formatting
- Run `flake8` for linting
- Run `mypy` for type checking

### Pull Request Process

1. Create a new branch for your feature/bugfix
2. Make your changes
3. Add tests for new functionality
4. Update documentation as needed
5. Run the test suite
6. Submit a pull request

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Reference issues and pull requests liberally

## Project Structure

### Core Components

- `dnaflex/models/`: Core analysis and prediction models
- `dnaflex/structure/`: Molecular structure handling
- `dnaflex/data/`: Data management and I/O
- `dnaflex/constants/`: Constant definitions

### Testing

- Write unit tests for new functionality
- Place tests in `tests/` directory
- Follow existing test patterns

## Documentation

- Update docstrings for any modified code
- Update API documentation for new features
- Add examples for new functionality
- Update README.md if needed

## Questions or Problems?

- Check existing issues
- Join discussions on GitHub
- Create new issues for bugs or feature requests

## License

By contributing, you agree that your contributions will be licensed under the MIT License.