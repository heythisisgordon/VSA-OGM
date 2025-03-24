"""
Vector operations for Vector Symbolic Architecture (VSA).

This module provides core VSA operations such as binding, power, inversion, and
normalization. These operations are optimized for batch processing and can be
accelerated with CUDA where available.
"""

import numpy as np
import torch
from typing import Union, List, Optional


def bind(vectors: Union[torch.Tensor, List[torch.Tensor]], 
         device: Optional[str] = None) -> torch.Tensor:
    """
    Bind multiple vectors into a single representation using circular convolution.
    
    Args:
        vectors: Either a tensor of shape (num_vectors, num_dimensions) or a list
                of tensors each with shape (num_dimensions)
        device: Device to perform operations on (defaults to vectors' device)
    
    Returns:
        A tensor of shape (num_dimensions) representing the bound vectors
    """
    if device is None and isinstance(vectors, torch.Tensor):
        device = vectors.device
    elif device is None:
        device = vectors[0].device
        
    if isinstance(vectors, list):
        # Convert list of vectors to a tensor
        batch_shape = (len(vectors), vectors[0].shape[-1])
        batch = torch.zeros(batch_shape, device=device)
        for i, vec in enumerate(vectors):
            batch[i, :] = vec
        vectors = batch

    # Perform binding in Fourier domain (circular convolution)
    vectors_fft = torch.fft.fft(vectors, dim=1)
    result_fft = torch.prod(vectors_fft, dim=0)
    result = torch.fft.ifft(result_fft).real
    
    return result


def power(vector: torch.Tensor, scalar: float, length_scale: float = 1.0) -> torch.Tensor:
    """
    Fractionally bind a vector to a continuous scalar using exponentiation in Fourier domain.
    
    Args:
        vector: A hypervector of shape (num_dimensions)
        scalar: The continuous value to exponentiate the vector with
        length_scale: Width of the kernel (controls sensitivity to scalar changes)
    
    Returns:
        A tensor of shape (num_dimensions) representing the fractionally bound vector
    """
    # Perform fractional binding in Fourier domain
    vector_fft = torch.fft.fft(vector)
    vector_fft_powered = vector_fft ** (scalar / length_scale)
    result = torch.fft.ifft(vector_fft_powered).real
    
    return result


def invert(vector: torch.Tensor) -> torch.Tensor:
    """
    Return the pseudo-inverse of a hypervector.
    
    Args:
        vector: A hypervector of shape (num_dimensions)
    
    Returns:
        A tensor of shape (num_dimensions) representing the inverted vector
    """
    # Invert by reversing indices (except for the first element)
    return vector[-torch.arange(vector.shape[0])]


def normalize(vector: torch.Tensor) -> torch.Tensor:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: A tensor of any shape
    
    Returns:
        The normalized tensor with the same shape
    """
    norm = torch.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


def make_unitary(num_dimensions: int, device: str = "cpu", eps: float = 1e-3) -> torch.Tensor:
    """
    Create a hyperdimensional vector of unitary length phasors.
    
    Args:
        num_dimensions: The dimensionality of the VSA vector
        device: Device to store the tensor on
        eps: Allowable variability in the phase of each phasor
    
    Returns:
        A tensor of shape (num_dimensions) with unitary phasors
    """
    # Generate random phases
    a = torch.rand((num_dimensions - 1) // 2)
    sign = np.random.choice((-1, +1), len(a))
    
    sign = torch.from_numpy(sign).to(device)
    a = a.to(device)

    # Create phases with constraints
    phi = sign * torch.pi * (eps + a * (1 - 2 * eps))

    # Ensure phases are within bounds
    assert torch.all(torch.abs(phi) >= torch.pi * eps)
    assert torch.all(torch.abs(phi) <= torch.pi * (1 - eps))

    # Create frequency domain representation
    fv = torch.zeros(num_dimensions, dtype=torch.complex64, device=device)
    fv[0] = 1
    fv[1:(num_dimensions + 1) // 2] = torch.cos(phi) + 1j * torch.sin(phi)
    fv[(num_dimensions // 2) + 1:] = torch.flip(torch.conj(fv[1:(num_dimensions + 1) // 2]), dims=[0])
    
    # Handle even-dimensional case
    if num_dimensions % 2 == 0:
        fv[num_dimensions // 2] = 1

    # Verify unitarity
    assert torch.allclose(torch.abs(fv), torch.ones(fv.shape, device=device))
    
    # Convert to time domain
    v = torch.fft.ifft(fv).real
    
    # Verify properties
    assert torch.allclose(torch.fft.fft(v), fv)
    assert torch.allclose(torch.linalg.norm(v), torch.tensor(1.0, device=device))

    return v


def batch_bind(vectors: torch.Tensor, axis_vectors: torch.Tensor, 
               values: torch.Tensor, length_scale: float = 1.0) -> torch.Tensor:
    """
    Efficiently bind multiple vectors with fractional powers in batch.
    
    Args:
        vectors: Tensor of shape (batch_size, num_dimensions)
        axis_vectors: Tensor of shape (num_axes, num_dimensions)
        values: Tensor of shape (batch_size, num_axes) containing values to encode
        length_scale: Width of the kernel
        
    Returns:
        Tensor of shape (batch_size, num_dimensions) with bound vectors
    """
    batch_size, num_dimensions = vectors.shape
    num_axes = axis_vectors.shape[0]
    
    # Convert to frequency domain
    vectors_fft = torch.fft.fft(vectors, dim=1)
    axis_fft = torch.fft.fft(axis_vectors, dim=1)
    
    # Prepare for broadcasting
    axis_fft = axis_fft.unsqueeze(0)  # [1, num_axes, num_dimensions]
    values = values.unsqueeze(2)      # [batch_size, num_axes, 1]
    
    # Apply fractional powers
    powered = axis_fft ** (values / length_scale)
    
    # Multiply across axes
    result_fft = torch.prod(powered, dim=1)  # [batch_size, num_dimensions]
    
    # Convert back to time domain
    result = torch.fft.ifft(result_fft).real
    
    return result


def similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector or batch of vectors
        v2: Second vector or batch of vectors
        
    Returns:
        Cosine similarity (scalar or tensor depending on input shapes)
    """
    # Normalize vectors
    v1_norm = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp(min=1e-8)
    v2_norm = v2 / torch.linalg.norm(v2, dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Calculate dot product
    return torch.sum(v1_norm * v2_norm, dim=-1)
