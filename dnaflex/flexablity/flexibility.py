"""DNA flexibility analysis using C++ accelerated parsers."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ..parsers.parsers_cpp import MSAProfile, MSAConverter
from ..models.dna_llm import BioLLM
from ..constants.atom_layouts import DNA_BASE_ATOMS

class DNAFlexibilityAnalyzer:
    """Analyzes DNA sequence flexibility using ML models and physics-based calculations."""
    
    def __init__(self, use_gpu: bool = False):
        """Initialize the analyzer.
        
        Args:
            use_gpu: Whether to use GPU acceleration for ML model inference
        """
        self.msa_profile = MSAProfile()
        self.msa_converter = MSAConverter()
        self.model = BioLLM(
            model_type="dna",
            embedding_size=128,
            hidden_size=256,
            num_heads=4,
            use_gpu=use_gpu
        )
        
    def analyze_sequence(self, sequence: str) -> Dict[str, Union[np.ndarray, float]]:
        """Analyze DNA sequence flexibility properties.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Dictionary containing:
            - bendability: Base-pair step bendability profile
            - twist_flexibility: Base-pair step twist flexibility profile
            - groove_width: Major and minor groove width estimates
            - conservation: Position-wise conservation scores if multiple sequences
            - stability: Base-pair step stability estimates
        """
        # Validate sequence
        sequence = sequence.upper()
        for base in sequence:
            if base not in 'ATGC':
                raise ValueError(f"Invalid base '{base}' in sequence")
        
        # Get structural predictions from ML model
        embeddings = self.model.embed_sequence(sequence)
        structure_features = self.model.predict_structure(embeddings)
        
        # Calculate flexibility metrics
        metrics = {
            'bendability': self._calc_bendability(sequence, structure_features),
            'twist_flexibility': self._calc_twist_flexibility(sequence, structure_features),
            'groove_width': self._calc_groove_geometry(sequence, structure_features),
            'stability': self._calc_stability(sequence)
        }
        
        # Add conservation if we have multiple sequences
        if self.msa_profile.get_sequence_count() > 1:
            metrics['conservation'] = self.msa_profile.get_conservation_scores()
        
        return metrics
    
    def add_homologous_sequence(self, sequence: str) -> None:
        """Add a homologous sequence for conservation analysis.
        
        Args:
            sequence: DNA sequence string
        """
        self.msa_profile.add_sequence(sequence)
        self.msa_profile.compute_profile()
    
    def _calc_bendability(self, sequence: str, features: Dict) -> np.ndarray:
        """Calculate DNA bendability profile."""
        # Consider both sequence-dependent parameters and structural predictions
        steps = len(sequence) - 1
        bendability = np.zeros(steps)
        
        for i in range(steps):
            # Get dinucleotide and its structural features
            dinuc = sequence[i:i+2]
            struct_contrib = features['bendability'][i]
            
            # Combine sequence and structure contributions
            bendability[i] = self._get_sequence_bendability(dinuc) * struct_contrib
            
        return bendability
    
    def _calc_twist_flexibility(self, sequence: str, features: Dict) -> np.ndarray:
        """Calculate DNA twist flexibility profile."""
        steps = len(sequence) - 1
        flexibility = np.zeros(steps)
        
        for i in range(steps):
            dinuc = sequence[i:i+2]
            struct_contrib = features['twist_flexibility'][i]
            flexibility[i] = self._get_sequence_twist_flex(dinuc) * struct_contrib
            
        return flexibility
    
    def _calc_groove_geometry(self, sequence: str, features: Dict) -> Dict[str, np.ndarray]:
        """Calculate major and minor groove geometries."""
        # Need 3 base pairs to define groove geometry
        n_windows = len(sequence) - 2
        major_groove = np.zeros(n_windows)
        minor_groove = np.zeros(n_windows)
        
        for i in range(n_windows):
            trinuc = sequence[i:i+3]
            struct_features = features['groove_geometry'][i]
            
            major_groove[i] = self._get_major_groove_width(trinuc) * struct_features['major']
            minor_groove[i] = self._get_minor_groove_width(trinuc) * struct_features['minor']
            
        return {
            'major_groove': major_groove,
            'minor_groove': minor_groove
        }
    
    def _calc_stability(self, sequence: str) -> np.ndarray:
        """Calculate base-pair step stability."""
        steps = len(sequence) - 1
        stability = np.zeros(steps)
        
        # Define stacking energy parameters (kcal/mol)
        stacking_energies = {
            'AA': -5.37, 'AT': -6.57, 'AG': -5.27, 'AC': -5.27,
            'TA': -6.57, 'TT': -5.37, 'TG': -5.27, 'TC': -5.27,
            'GA': -5.27, 'GT': -5.27, 'GG': -5.37, 'GC': -8.26,
            'CA': -5.27, 'CT': -5.27, 'CG': -8.26, 'CC': -5.37
        }
        
        for i in range(steps):
            dinuc = sequence[i:i+2]
            stability[i] = -stacking_energies.get(dinuc, 0)  # Negative because higher energy = less stable
            
        return stability
    
    @staticmethod
    def _get_sequence_bendability(dinuc: str) -> float:
        """Get sequence-dependent bendability parameter."""
        # Parameters from experimental studies
        bendability = {
            'AA': 1.2, 'AT': 1.0, 'AG': 0.9, 'AC': 0.8,
            'TA': 1.3, 'TT': 1.2, 'TG': 0.8, 'TC': 0.7,
            'GA': 0.9, 'GT': 0.8, 'GG': 1.1, 'GC': 0.6,
            'CA': 0.8, 'CT': 0.7, 'CG': 0.6, 'CC': 1.1
        }
        return bendability.get(dinuc, 1.0)
    
    @staticmethod
    def _get_sequence_twist_flex(dinuc: str) -> float:
        """Get sequence-dependent twist flexibility parameter."""
        # Parameters derived from MD simulations
        twist_flex = {
            'AA': 1.1, 'AT': 1.2, 'AG': 0.9, 'AC': 0.8,
            'TA': 1.3, 'TT': 1.1, 'TG': 0.8, 'TC': 0.7,
            'GA': 0.9, 'GT': 0.8, 'GG': 1.0, 'GC': 0.7,
            'CA': 0.8, 'CT': 0.7, 'CG': 0.7, 'CC': 1.0
        }
        return twist_flex.get(dinuc, 1.0)
    
    @staticmethod
    def _get_major_groove_width(trinuc: str) -> float:
        """Estimate sequence-dependent major groove width."""
        # GC content affects major groove width
        gc_count = trinuc.count('G') + trinuc.count('C')
        return 1.0 + 0.1 * gc_count  # Base width + GC contribution
    
    @staticmethod
    def _get_minor_groove_width(trinuc: str) -> float:
        """Estimate sequence-dependent minor groove width."""
        # AT content affects minor groove width
        at_count = trinuc.count('A') + trinuc.count('T')
        return 1.0 + 0.1 * at_count  # Base width + AT contribution