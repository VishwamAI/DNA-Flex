"""Data loading utilities for DNA-Flex."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import numpy as np
from Bio import SeqIO

class DataLoader:
    """Load and parse DNA-related data from various formats."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize data loader.
        
        Args:
            cache_dir: Optional directory for caching loaded data
        """
        self.cache_dir = cache_dir or Path.home() / '.dnaflex' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_fasta(self, file_path: Union[str, Path]) -> Dict[str, str]:
        """Load DNA sequences from FASTA file.
        
        Args:
            file_path: Path to FASTA file
            
        Returns:
            Dictionary mapping sequence IDs to sequences
        """
        sequences = {}
        for record in SeqIO.parse(str(file_path), "fasta"):
            sequences[record.id] = str(record.seq)
        return sequences
        
    def load_pdb(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load DNA structure from PDB file.
        
        Args:
            file_path: Path to PDB file
            
        Returns:
            Dictionary containing parsed structure data
        """
        from Bio.PDB import PDBParser
        
        parser = PDBParser()
        structure = parser.get_structure('dna', str(file_path))
        
        # Extract relevant data
        data = {
            'chains': {},
            'metadata': {
                'id': structure.id,
                'header': structure.header
            }
        }
        
        for model in structure:
            for chain in model:
                chain_data = []
                for residue in chain:
                    res_data = {
                        'id': residue.id[1],
                        'name': residue.resname,
                        'atoms': {}
                    }
                    for atom in residue:
                        res_data['atoms'][atom.name] = {
                            'coords': atom.coord.tolist(),
                            'element': atom.element,
                            'bfactor': atom.bfactor,
                            'occupancy': atom.occupancy
                        }
                    chain_data.append(res_data)
                data['chains'][chain.id] = chain_data
                
        return data
        
    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(file_path) as f:
            return json.load(f)
            
    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save data to JSON file.
        
        Args:
            data: Data to save
            file_path: Output file path
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_numpy(self, file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load numeric data from NumPy file.
        
        Args:
            file_path: Path to .npz file
            
        Returns:
            Dictionary of arrays
        """
        return dict(np.load(file_path))
        
    def save_numpy(self, data: Dict[str, np.ndarray], file_path: Union[str, Path]) -> None:
        """Save numeric data to NumPy file.
        
        Args:
            data: Dictionary of arrays to save
            file_path: Output file path
        """
        np.savez_compressed(file_path, **data)
        
    def load_sequence_db(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load sequence database in custom format.
        
        Args:
            file_path: Path to database file
            
        Returns:
            Dictionary containing sequence database
        """
        with open(file_path) as f:
            lines = f.readlines()
            
        database = {
            'sequences': {},
            'metadata': {}
        }
        
        current_seq = None
        current_meta = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_seq is not None:
                    seq_id = current_meta.get('id', f'seq_{len(database["sequences"])}')
                    database['sequences'][seq_id] = {
                        'sequence': current_seq,
                        'metadata': current_meta
                    }
                
                # Start new sequence
                current_seq = ''
                current_meta = {}
                
                # Parse header
                header = line[1:].split('|')
                if len(header) >= 2:
                    current_meta['id'] = header[0].strip()
                    current_meta['description'] = header[1].strip()
                else:
                    current_meta['id'] = header[0].strip()
                    
            elif line.startswith('#'):
                # Parse metadata
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    current_meta[key.strip()] = value.strip()
            else:
                # Sequence data
                current_seq += line.strip()
                
        # Save last sequence
        if current_seq is not None:
            seq_id = current_meta.get('id', f'seq_{len(database["sequences"])}')
            database['sequences'][seq_id] = {
                'sequence': current_seq,
                'metadata': current_meta
            }
            
        return database