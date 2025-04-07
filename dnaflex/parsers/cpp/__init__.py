"""
DNA-Flex C++ accelerated parsers module.
Provides efficient implementations of CIF dictionary parsing, FASTA file iteration,
and MSA (Multiple Sequence Alignment) format conversion.
"""

from .parsers_cpp import (
    CifEntry,
    CifDictionary,
    FastaEntry,
    FastaIterator,
    MSAConverter
)

__all__ = [
    'CifEntry',
    'CifDictionary',
    'FastaEntry',
    'FastaIterator',
    'MSAConverter'
]