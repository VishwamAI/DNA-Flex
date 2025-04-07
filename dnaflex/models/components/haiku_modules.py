"""Neural network components implemented in Haiku."""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Tuple

class Linear(hk.Module):
    """Linear layer with configurable initialization."""
    
    def __init__(self,
                 output_size: int,
                 with_bias: bool = True,
                 w_init: Optional[hk.initializers.Initializer] = None,
                 b_init: Optional[hk.initializers.Initializer] = None,
                 precision: Optional[jax.lax.Precision] = None,
                 name: Optional[str] = None):
        """Initialize linear layer.
        
        Args:
            output_size: Number of output features
            with_bias: Whether to add a bias to the output
            w_init: Weight initializer
            b_init: Bias initializer
            precision: Numerical precision policy
            name: Name of the module
        """
        super().__init__(name=name)
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init or hk.initializers.VarianceScaling(1.0)
        self.b_init = b_init or hk.initializers.Constant(0.0)
        self.precision = precision
        
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply linear transformation."""
        input_shape = inputs.shape
        
        if not input_shape:
            raise ValueError("Input must not be scalar.")
            
        input_size = input_shape[-1]
        dtype = inputs.dtype
        
        w = hk.get_parameter("w",
                           shape=[input_size, self.output_size],
                           dtype=dtype,
                           init=self.w_init)
                           
        out = jnp.dot(inputs, w, precision=self.precision)
        
        if self.with_bias:
            b = hk.get_parameter("b",
                              shape=[self.output_size],
                              dtype=dtype,
                              init=self.b_init)
            out = out + b
            
        return out

class LayerNorm(hk.Module):
    """Layer normalization module."""
    
    def __init__(self, 
                 axis: int = -1,
                 eps: float = 1e-5,
                 name: Optional[str] = None):
        """Initialize layer normalization.
        
        Args:
            axis: Axis to normalize over
            eps: Small constant for numerical stability
            name: Module name
        """
        super().__init__(name=name)
        self.axis = axis
        self.eps = eps
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply layer normalization.
        
        Args:
            x: Input array
            
        Returns:
            Normalized array
        """
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        variance = jnp.var(x, axis=self.axis, keepdims=True)
        scale = hk.get_parameter("scale", x.shape[-1:], init=jnp.ones)
        offset = hk.get_parameter("offset", x.shape[-1:], init=jnp.zeros)
        
        inv = scale * jax.lax.rsqrt(variance + self.eps)
        return (x - mean) * inv + offset

class MultiHeadAttention(hk.Module):
    """Multi-head attention module."""
    
    def __init__(self,
                 num_heads: int,
                 key_size: int,
                 model_size: Optional[int] = None,
                 w_init: Optional[hk.initializers.Initializer] = None,
                 value_size: Optional[int] = None,
                 name: Optional[str] = None):
        """Initialize multi-head attention.
        
        Args:
            num_heads: Number of attention heads
            key_size: Size of each attention head
            model_size: Overall model dimensionality (defaults to key_size * num_heads)
            w_init: Weight initializer
            value_size: Size of value heads (defaults to key_size)
            name: Module name
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(2.0)
        self.w_init = w_init
        
    def __call__(self,
                 query: jnp.ndarray,
                 key: jnp.ndarray,
                 value: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply multi-head attention.
        
        Args:
            query: Query vectors [B, T, D]
            key: Key vectors [B, S, D] 
            value: Value vectors [B, S, D]
            mask: Optional attention mask [B, H, T, S]
            
        Returns:
            Output vectors [B, T, D]
            
        Where B=batch size, T=target sequence length, S=source sequence length,
        D=model dimension, H=number of heads.
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        def reshape_for_heads(x: jnp.ndarray) -> jnp.ndarray:
            """Split channels into attention heads."""
            return x.reshape(batch_size, -1, self.num_heads, self.key_size)
            
        # Project inputs to queries, keys and values
        query_heads = self._project("query", query, self.key_size * self.num_heads)
        key_heads = self._project("key", key, self.key_size * self.num_heads)  
        value_heads = self._project("value", value, self.value_size * self.num_heads)
        
        # Reshape to split channels into heads
        query_heads = reshape_for_heads(query_heads)
        key_heads = reshape_for_heads(key_heads)
        value_heads = reshape_for_heads(value_heads)
        
        # Calculate attention weights
        depth = self.key_size
        attn_logits = jnp.einsum("bthd,bshd->bhts", query_heads, key_heads)
        attn_logits = attn_logits / jnp.sqrt(depth).astype(key_heads.dtype)
        
        # Apply mask if provided
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -1e30)
            
        # Get attention weights
        attn_weights = jax.nn.softmax(attn_logits)
        
        # Apply attention to values
        attn_vec = jnp.einsum("bhts,bshd->bthd", attn_weights, value_heads)
        
        # Merge heads
        attn_vec = attn_vec.reshape(batch_size, seq_len, -1)
        
        # Final projection
        return self._project("output", attn_vec, self.model_size)
        
    def _project(self,
                name: str,
                x: jnp.ndarray,
                output_size: int) -> jnp.ndarray:
        """Project input vectors to output size."""
        return hk.Linear(output_size, w_init=self.w_init, name=name)(x)

class FeedForward(hk.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self,
                 hidden_size: int,
                 output_size: int,
                 dropout_rate: float = 0.1,
                 name: Optional[str] = None):
        """Initialize feed-forward network.
        
        Args:
            hidden_size: Size of hidden layer
            output_size: Size of output
            dropout_rate: Dropout probability
            name: Module name
        """
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
    def __call__(self,
                 x: jnp.ndarray,
                 is_training: bool = True) -> jnp.ndarray:
        """Apply feed-forward network.
        
        Args:
            x: Input array
            is_training: Whether in training mode
            
        Returns:
            Transformed array
        """
        # First dense layer with GELU activation
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.gelu(x)
        
        # Dropout
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
            
        # Output projection
        return hk.Linear(self.output_size)(x)

class TransformerBlock(hk.Module):
    """Transformer block combining attention and feed-forward layers."""
    
    def __init__(self,
                 num_heads: int,
                 key_size: int,
                 ff_size: int,
                 dropout_rate: float = 0.1,
                 name: Optional[str] = None):
        """Initialize transformer block.
        
        Args:
            num_heads: Number of attention heads
            key_size: Size of attention heads
            ff_size: Size of feed-forward hidden layer
            dropout_rate: Dropout probability
            name: Module name
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.ff_size = ff_size
        self.dropout_rate = dropout_rate
        
    def __call__(self,
                 inputs: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 is_training: bool = True) -> jnp.ndarray:
        """Apply transformer block.
        
        Args:
            inputs: Input array
            mask: Optional attention mask
            is_training: Whether in training mode
            
        Returns:
            Transformed array
        """
        # Multi-head self-attention
        x = LayerNorm()(inputs)
        attention_out = MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size)(x, x, x, mask)
            
        if is_training:
            attention_out = hk.dropout(
                hk.next_rng_key(), self.dropout_rate, attention_out)
        x = inputs + attention_out
        
        # Feed-forward network
        y = LayerNorm()(x)
        ff_out = FeedForward(
            hidden_size=self.ff_size,
            output_size=inputs.shape[-1],
            dropout_rate=self.dropout_rate)(y, is_training)
            
        return x + ff_out