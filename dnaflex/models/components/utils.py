"""Utility functions for DNA-Flex components."""

import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Tuple, Union
import jax
def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between 3D coordinates.
    
    Args:
        coords: Array of shape (N, 3) containing 3D coordinates
        
    Returns:
        Array of shape (N, N) containing pairwise distances
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def mask_mean(
    mask: jnp.ndarray,
    value: jnp.ndarray,
    axis: Union[int, Tuple[int, ...]], 
    eps: float = 1e-10
) -> jnp.ndarray:
    """Compute mean of values with masking.
    
    Args:
        mask: Binary mask array
        value: Array of values to average
        axis: Axis or axes along which to compute mean
        eps: Small value to prevent division by zero
        
    Returns:
        Masked mean along specified axes
    """
    mask = mask.astype(value.dtype)
    total = jnp.sum(mask * value, axis=axis)
    count = jnp.sum(mask, axis=axis)
    return total / (count + eps)

def grid_sample(key: jnp.ndarray, shape: Tuple[int, ...], k: int) -> jnp.ndarray:
    """Sample k evenly spaced indices from grid.
    
    Args:
        key: JAX random key
        shape: Shape of grid to sample from
        k: Number of samples to take
        
    Returns:
        Array of k evenly spaced indices
    """
    size = np.prod(shape)
    if k >= size:
        return jnp.arange(size)
        
    spacing = size / k
    base = jnp.arange(k) * spacing
    noise = jax.random.uniform(key, (k,)) * spacing
    indices = jnp.int32(base + noise) % size
    return indices

def split_and_pad(
    array: np.ndarray,
    chunk_size: int,
    pad_value: float = 0.0
) -> List[np.ndarray]:
    """Split array into chunks with padding.
    
    Args:
        array: Input array to split
        chunk_size: Size of each chunk
        pad_value: Value to use for padding
        
    Returns:
        List of padded chunks
    """
    chunks = []
    for i in range(0, len(array), chunk_size):
        chunk = array[i:i + chunk_size]
        if len(chunk) < chunk_size:
            pad_width = [(0, chunk_size - len(chunk))] + [(0, 0)] * (chunk.ndim - 1)
            chunk = np.pad(chunk, pad_width, constant_values=pad_value)
        chunks.append(chunk)
    return chunks

def normalize_vector(v: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Normalize a vector or batch of vectors.
    
    Args:
        v: Vector(s) to normalize
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized vector(s)
    """
    norm = jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True))
    return v / (norm + eps)

def rotation_matrix_from_vectors(vec1: jnp.ndarray, vec2: jnp.ndarray) -> jnp.ndarray:
    """Compute rotation matrix that aligns two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        3x3 rotation matrix
    """
    v1 = normalize_vector(vec1)
    v2 = normalize_vector(vec2)
    
    cross = jnp.cross(v1, v2)
    cos_angle = jnp.dot(v1, v2)
    
    if jnp.abs(cos_angle + 1.0) < 1e-6:
        # Vectors are antiparallel
        return -jnp.eye(3)
        
    k = 1.0 / (1.0 + cos_angle)
    return jnp.eye(3) + cross + k * jnp.outer(cross, cross)