"""Feature batch processing module."""

from typing import Dict, Any, Optional, NamedTuple
import jax.numpy as jnp
import numpy as np
from dnaflex.models import features

class MSAFeatures(NamedTuple):
    """MSA features for a batch."""
    rows: jnp.ndarray  # [num_seq, num_res] MSA sequence tokens
    deletion_matrix: jnp.ndarray  # [num_seq, num_res] Number of deletions in MSA
    profile: jnp.ndarray  # [num_res, num_residue_types] AA profile from MSA
    deletion_mean: jnp.ndarray  # [num_res] Mean deletion value for each position

class TokenFeatures(NamedTuple):
    """Token-level features."""
    aatype: jnp.ndarray  # [num_res] Residue types
    atom_mask: jnp.ndarray  # [num_res, num_atom_type] Mask for atom positions
    residue_index: jnp.ndarray  # [num_res] Residue indices
    chain_index: jnp.ndarray  # [num_res] Chain indices
    seq_mask: jnp.ndarray  # [num_res] Sequence mask

class StructureFeatures(NamedTuple):
    """Structure-level features."""
    atom_positions: jnp.ndarray  # [num_res, num_atom_type, 3] Atom coordinates
    frames: jnp.ndarray  # [num_res, 7] Reference frames
    mask: jnp.ndarray  # [num_res] Structure validity mask
    resolution: jnp.ndarray  # [] Structure resolution
    element: jnp.ndarray  # [num_res, num_atom_type] Atom element types

class Batch:
    """A batch of features for model input."""
    
    def __init__(
        self,
        msa: MSAFeatures,
        token_features: TokenFeatures,
        ref_structure: Optional[StructureFeatures] = None,
        **kwargs
    ):
        """Initialize batch with features.
        
        Args:
            msa: MSA features
            token_features: Token-level features 
            ref_structure: Optional reference structure features
            **kwargs: Additional features
        """
        self.msa = msa
        self.token_features = token_features
        self.ref_structure = ref_structure
        self._additional_features = kwargs
        
    def __getattr__(self, name: str) -> Any:
        if name in self._additional_features:
            return self._additional_features[name]
        raise AttributeError(f"'Batch' object has no attribute '{name}'")
        
    @classmethod
    def from_features(
        cls,
        feature_dict: Dict[str, jnp.ndarray]
    ) -> 'Batch':
        """Create batch from feature dictionary.
        
        Args:
            feature_dict: Dictionary of features
            
        Returns:
            New Batch instance
        """
        # Extract MSA features
        msa = MSAFeatures(
            rows=feature_dict['msa'],
            deletion_matrix=feature_dict['deletion_matrix'],
            profile=feature_dict['msa_profile'],
            deletion_mean=feature_dict['deletion_mean']
        )
        
        # Extract token features
        token_features = TokenFeatures(
            aatype=feature_dict['aatype'],
            atom_mask=feature_dict['atom_mask'],
            residue_index=feature_dict['residue_index'],
            chain_index=feature_dict['chain_index'],
            seq_mask=feature_dict['seq_mask']
        )
        
        # Extract optional structure features
        if 'atom_positions' in feature_dict:
            ref_structure = StructureFeatures(
                atom_positions=feature_dict['atom_positions'],
                frames=feature_dict['frames'],
                mask=feature_dict['structure_mask'],
                resolution=feature_dict['resolution'],
                element=feature_dict['element']
            )
        else:
            ref_structure = None
            
        # Get any additional features
        additional = {
            k: v for k, v in feature_dict.items()
            if k not in {
                'msa', 'deletion_matrix', 'msa_profile', 'deletion_mean',
                'aatype', 'atom_mask', 'residue_index', 'chain_index', 'seq_mask',
                'atom_positions', 'frames', 'structure_mask', 'resolution', 'element'
            }
        }
        
        return cls(msa, token_features, ref_structure, **additional)
        
    def to_dict(self) -> Dict[str, jnp.ndarray]:
        """Convert batch to feature dictionary.
        
        Returns:
            Dictionary of features
        """
        feature_dict = {
            # MSA features
            'msa': self.msa.rows,
            'deletion_matrix': self.msa.deletion_matrix,
            'msa_profile': self.msa.profile,
            'deletion_mean': self.msa.deletion_mean,
            
            # Token features
            'aatype': self.token_features.aatype,
            'atom_mask': self.token_features.atom_mask,
            'residue_index': self.token_features.residue_index,
            'chain_index': self.token_features.chain_index,
            'seq_mask': self.token_features.seq_mask,
        }
        
        # Add structure features if present
        if self.ref_structure is not None:
            feature_dict.update({
                'atom_positions': self.ref_structure.atom_positions,
                'frames': self.ref_structure.frames,
                'structure_mask': self.ref_structure.mask,
                'resolution': self.ref_structure.resolution,
                'element': self.ref_structure.element
            })
            
        # Add any additional features
        feature_dict.update(self._additional_features)
        
        return feature_dict
        
    def slice_batch(self, index: jnp.ndarray) -> 'Batch':
        """Create new batch by slicing at given indices.
        
        Args:
            index: Indices to select
            
        Returns:
            New Batch with selected indices
        """
        msa = MSAFeatures(
            rows=self.msa.rows[:, index],
            deletion_matrix=self.msa.deletion_matrix[:, index],
            profile=self.msa.profile[index],
            deletion_mean=self.msa.deletion_mean[index]
        )
        
        token_features = TokenFeatures(
            aatype=self.token_features.aatype[index],
            atom_mask=self.token_features.atom_mask[index],
            residue_index=self.token_features.residue_index[index],
            chain_index=self.token_features.chain_index[index],
            seq_mask=self.token_features.seq_mask[index]
        )
        
        ref_structure = None
        if self.ref_structure is not None:
            ref_structure = StructureFeatures(
                atom_positions=self.ref_structure.atom_positions[index],
                frames=self.ref_structure.frames[index],
                mask=self.ref_structure.mask[index],
                resolution=self.ref_structure.resolution,
                element=self.ref_structure.element[index]
            )
            
        additional = {
            k: v[index] if isinstance(v, (np.ndarray, jnp.ndarray)) else v
            for k, v in self._additional_features.items()
        }
        
        return Batch(msa, token_features, ref_structure, **additional)
        
    def slice_msa(self, index: jnp.ndarray) -> 'Batch':
        """Create new batch by slicing MSA sequences.
        
        Args:
            index: Indices of MSA sequences to select
            
        Returns:
            New Batch with selected MSA sequences
        """
        msa = MSAFeatures(
            rows=self.msa.rows[index],
            deletion_matrix=self.msa.deletion_matrix[index],
            profile=self.msa.profile,  # Profile is per-position
            deletion_mean=self.msa.deletion_mean  # Mean is per-position
        )
        
        return Batch(
            msa,
            self.token_features,
            self.ref_structure,
            **self._additional_features
        )