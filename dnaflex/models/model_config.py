"""Model configuration module."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import jax.numpy as jnp

@dataclass
class GlobalConfig:
    """Global model configuration."""
    
    # Model architecture
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    
    # Input parameters
    max_sequence_length: int = 1024
    vocab_size: int = 5  # ACGTN
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_gradient_norm: float = 1.0
    
    # Model specific settings
    use_relative_attention: bool = True
    use_rotary_embeddings: bool = True
    num_position_embeddings: int = 1024
    
    # Feature dimensions
    num_sequence_features: int = 5
    num_structure_features: int = 3
    num_evolutionary_features: int = 21
    
    # Initialization
    initializer_range: float = 0.02
    bfloat16: str = 'inference'  # Options: 'all', 'inference', 'none'

@dataclass
class ModelConfig:
    """Configuration for DNA sequence models."""
    
    # Sequence processing
    max_seq_length: int = 1024
    min_seq_length: int = 20
    vocab_size: int = 5
    pad_token_id: int = 4
    
    # Model architecture
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    hidden_act: str = "gelu"
    
    # Attention
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    type_vocab_size: int = 2
    
    # Layer specific
    layer_norm_eps: float = 1e-12
    gradient_checkpointing: bool = False
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Features
    use_sequence_features: bool = True
    use_structure_features: bool = True
    use_evolutionary_features: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: getattr(self, key)
            for key in self.__annotations__
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**{
            key: value
            for key, value in config_dict.items()
            if key in cls.__annotations__
        })