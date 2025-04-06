"""Data providers for DNA-Flex."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import requests
from Bio import Entrez
from .cache import DataCache
from .loader import DataLoader

class SequenceProvider:
    """Provider for DNA sequence data."""
    
    def __init__(self, email: str, cache_dir: Optional[Path] = None):
        """Initialize sequence provider.
        
        Args:
            email: Email for NCBI services
            cache_dir: Optional cache directory
        """
        self.email = email
        Entrez.email = email
        self.cache = DataCache(cache_dir)
        self.loader = DataLoader(cache_dir)
        
    def fetch_from_ncbi(self, accession: str) -> Dict[str, Any]:
        """Fetch sequence from NCBI.
        
        Args:
            accession: NCBI accession number
            
        Returns:
            Dictionary with sequence data
        """
        cache_key = f"ncbi_{accession}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        # Fetch from NCBI
        handle = Entrez.efetch(
            db="nucleotide",
            id=accession,
            rettype="gb",
            retmode="text"
        )
        record = handle.read()
        handle.close()
        
        # Parse and structure the data
        result = {
            'accession': accession,
            'source': 'NCBI',
            'data': record
        }
        
        self.cache.set(cache_key, result)
        return result
        
    def fetch_from_ensembl(self, identifier: str) -> Dict[str, Any]:
        """Fetch sequence from Ensembl.
        
        Args:
            identifier: Ensembl identifier
            
        Returns:
            Dictionary with sequence data
        """
        cache_key = f"ensembl_{identifier}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        # Ensembl REST API endpoint
        base_url = "https://rest.ensembl.org"
        endpoint = f"/sequence/id/{identifier}"
        
        response = requests.get(
            f"{base_url}{endpoint}",
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = {
            'identifier': identifier,
            'source': 'Ensembl',
            'data': response.json()
        }
        
        self.cache.set(cache_key, result)
        return result
        
    def load_local_database(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load sequences from local database.
        
        Args:
            file_path: Path to sequence database file
            
        Returns:
            Dictionary with loaded sequences
        """
        return self.loader.load_sequence_db(file_path)

class StructureProvider:
    """Provider for DNA structure data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize structure provider.
        
        Args:
            cache_dir: Optional cache directory
        """
        self.cache = DataCache(cache_dir)
        self.loader = DataLoader(cache_dir)
        
    def fetch_from_pdb(self, pdb_id: str) -> Dict[str, Any]:
        """Fetch structure from PDB.
        
        Args:
            pdb_id: PDB identifier
            
        Returns:
            Dictionary with structure data
        """
        cache_key = f"pdb_{pdb_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        # PDB REST API endpoint
        base_url = "https://files.rcsb.org/download"
        pdb_url = f"{base_url}/{pdb_id}.pdb"
        
        response = requests.get(pdb_url)
        response.raise_for_status()
        
        # Save PDB file temporarily
        temp_file = self.cache.cache_dir / f"{pdb_id}.pdb"
        with open(temp_file, 'w') as f:
            f.write(response.text)
            
        # Parse structure
        try:
            structure_data = self.loader.load_pdb(temp_file)
            result = {
                'pdb_id': pdb_id,
                'source': 'PDB',
                'data': structure_data
            }
            
            self.cache.set(cache_key, result)
            return result
        finally:
            # Clean up temporary file
            temp_file.unlink()
            
    def load_local_structure(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load structure from local PDB file.
        
        Args:
            file_path: Path to PDB file
            
        Returns:
            Dictionary with structure data
        """
        return self.loader.load_pdb(file_path)