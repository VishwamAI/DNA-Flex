"""Central data management for DNA-Flex."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import shutil
from datetime import datetime

from .cache import DataCache
from .loader import DataLoader
from .providers import SequenceProvider, StructureProvider

class DataManager:
    """Central manager for DNA-Flex data operations."""
    
    def __init__(self, 
                 data_dir: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 email: str = 'dnaflex@example.com'):
        """Initialize data manager.
        
        Args:
            data_dir: Base directory for data storage
            cache_dir: Directory for cache storage
            email: Email for external services
        """
        self.data_dir = data_dir or Path.home() / '.dnaflex' / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cache = DataCache(cache_dir)
        self.loader = DataLoader(cache_dir)
        self.sequence_provider = SequenceProvider(email, cache_dir)
        self.structure_provider = StructureProvider(cache_dir)
        
        # Create data subdirectories
        self.sequences_dir = self.data_dir / 'sequences'
        self.structures_dir = self.data_dir / 'structures'
        self.results_dir = self.data_dir / 'results'
        
        for directory in [self.sequences_dir, self.structures_dir, self.results_dir]:
            directory.mkdir(exist_ok=True)
            
    def get_sequence(self, identifier: str, source: str = 'local') -> Dict[str, Any]:
        """Get sequence data from specified source.
        
        Args:
            identifier: Sequence identifier
            source: Data source ('local', 'ncbi', or 'ensembl')
            
        Returns:
            Dictionary containing sequence data
        """
        if source == 'local':
            file_path = self.sequences_dir / f"{identifier}.fasta"
            return self.loader.load_fasta(file_path)
        elif source == 'ncbi':
            return self.sequence_provider.fetch_from_ncbi(identifier)
        elif source == 'ensembl':
            return self.sequence_provider.fetch_from_ensembl(identifier)
        else:
            raise ValueError(f"Unknown source: {source}")
            
    def get_structure(self, identifier: str, source: str = 'local') -> Dict[str, Any]:
        """Get structure data from specified source.
        
        Args:
            identifier: Structure identifier
            source: Data source ('local' or 'pdb')
            
        Returns:
            Dictionary containing structure data
        """
        if source == 'local':
            file_path = self.structures_dir / f"{identifier}.pdb"
            return self.loader.load_pdb(file_path)
        elif source == 'pdb':
            return self.structure_provider.fetch_from_pdb(identifier)
        else:
            raise ValueError(f"Unknown source: {source}")
            
    def save_sequence(self, sequence_data: Dict[str, Any], 
                     identifier: str, format: str = 'fasta') -> Path:
        """Save sequence data to local storage.
        
        Args:
            sequence_data: Sequence data to save
            identifier: Sequence identifier
            format: Output format ('fasta' or 'json')
            
        Returns:
            Path to saved file
        """
        if format == 'fasta':
            output_path = self.sequences_dir / f"{identifier}.fasta"
            # Convert dictionary format to FASTA
            with open(output_path, 'w') as f:
                f.write(f">{identifier}\n")
                f.write(sequence_data.get('sequence', ''))
        elif format == 'json':
            output_path = self.sequences_dir / f"{identifier}.json"
            self.loader.save_json(sequence_data, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return output_path
        
    def save_structure(self, structure_data: Dict[str, Any], 
                      identifier: str) -> Path:
        """Save structure data to local storage.
        
        Args:
            structure_data: Structure data to save
            identifier: Structure identifier
            
        Returns:
            Path to saved file
        """
        output_path = self.structures_dir / f"{identifier}.json"
        self.loader.save_json(structure_data, output_path)
        return output_path
        
    def save_results(self, results: Dict[str, Any], 
                    analysis_type: str, identifier: str) -> Path:
        """Save analysis results.
        
        Args:
            results: Analysis results to save
            analysis_type: Type of analysis
            identifier: Result identifier
            
        Returns:
            Path to saved results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{identifier}_{analysis_type}_{timestamp}.json"
        output_path = self.results_dir / filename
        
        self.loader.save_json(results, output_path)
        return output_path
        
    def get_results(self, identifier: str = None, 
                   analysis_type: str = None) -> List[Dict[str, Any]]:
        """Get analysis results.
        
        Args:
            identifier: Optional result identifier filter
            analysis_type: Optional analysis type filter
            
        Returns:
            List of matching results
        """
        results = []
        
        for file_path in self.results_dir.glob("*.json"):
            try:
                result = self.loader.load_json(file_path)
                
                # Apply filters if specified
                if identifier and identifier not in file_path.stem:
                    continue
                if analysis_type and analysis_type not in file_path.stem:
                    continue
                    
                results.append({
                    'path': file_path,
                    'data': result
                })
            except Exception:
                continue
                
        return results
        
    def cleanup(self, older_than_days: int = 30) -> None:
        """Clean up old data and cache files.
        
        Args:
            older_than_days: Remove files older than this many days
        """
        # Clean up cache
        self.cache.cleanup()
        
        # Clean up old results
        cutoff = datetime.now().timestamp() - (older_than_days * 86400)
        
        for directory in [self.sequences_dir, self.structures_dir, self.results_dir]:
            for file_path in directory.glob("*"):
                if file_path.stat().st_mtime < cutoff:
                    file_path.unlink()
                    
    def export_data(self, export_dir: Union[str, Path],
                   include_cache: bool = False) -> None:
        """Export all data to specified directory.
        
        Args:
            export_dir: Directory to export to
            include_cache: Whether to include cache files
        """
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export data directories
        for src_dir in [self.sequences_dir, self.structures_dir, self.results_dir]:
            dst_dir = export_dir / src_dir.name
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            
        if include_cache:
            cache_dst = export_dir / 'cache'
            shutil.copytree(self.cache.cache_dir, cache_dst, dirs_exist_ok=True)
            
    def import_data(self, import_dir: Union[str, Path],
                   include_cache: bool = False) -> None:
        """Import data from specified directory.
        
        Args:
            import_dir: Directory to import from
            include_cache: Whether to import cache files
        """
        import_dir = Path(import_dir)
        
        # Import data directories
        for src_dir in ['sequences', 'structures', 'results']:
            src_path = import_dir / src_dir
            if src_path.exists():
                dst_path = self.data_dir / src_dir
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                
        if include_cache:
            cache_src = import_dir / 'cache'
            if cache_src.exists():
                shutil.copytree(cache_src, self.cache.cache_dir, dirs_exist_ok=True)