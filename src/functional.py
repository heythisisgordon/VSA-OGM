"""Vector operations for VSA-OGM."""

import numpy as np
import torch
from typing import Union, List

def bind_batch(vectors_list: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Bind a batch of vectors using element-wise multiplication in the frequency domain.
    
    Args:
        vectors_list: A list of tensors, each of shape [batch_size, vsa_dimensions]
        device: The device to perform the operation on
        
    Returns:
        A tensor of shape [batch_size, vsa_dimensions] containing the bound vectors
    """
    # Convert to frequency domain
    vectors_fd = [torch.fft.fft(v) for v in vectors_list]
    
    # Multiply in frequency domain (element-wise)
    result_fd = vectors_fd[0]
    for v in vectors_fd[1:]:
        result_fd = result_fd * v
    
    # Convert back to time domain
    result = torch.fft.ifft(result_fd).real
    
    return result

def bind(vs: Union[torch.Tensor, List[torch.Tensor]], device: torch.device) -> torch.Tensor:
    """
    Bind all argued vectors into a single representation with circular convolution.

    Args:
        vs: The vectors to bind together. Can be either:
            - A tensor of shape [num_vectors, num_vsa_dimensions]
            - A list of tensors, each of shape [num_vsa_dimensions]
        device: The device to perform operations on

    Returns:
        A tensor of shape [num_vsa_dimensions] representing the bound vectors
    """
    if isinstance(vs, list):
        ubvm_shape = (len(vs), vs[0].shape[-1])
        ubvm = torch.zeros(ubvm_shape, device=device)
        for i, value in enumerate(vs):
            ubvm[i, :] = value

        vs = ubvm

    vs = torch.fft.fft(vs, dim=1)
    vs = torch.prod(vs, dim=0)
    vs = torch.fft.ifft(vs).real
    return vs
    
def invert(ssp: torch.Tensor) -> torch.Tensor:
    """
    Return the pseudo inverse of the argued hypervector.

    Args:
        ssp: A hypervector of shape [num_vsa_dimensions]

    Returns:
        The pseudo inverse of the input vector
    """
    return ssp[-torch.arange(ssp.shape[0])]

def power(ssp: torch.Tensor, scalar: Union[float, torch.Tensor], length_scale: float = 1.0) -> torch.Tensor:
    """
    Fractionally bind hypervectors to continuous scalars with exponentiation in the Fourier domain.

    Args:
        ssp: A hypervector of shape [num_vsa_dimensions] or [batch_size, num_vsa_dimensions]
        scalar: The continuous value(s) to exponentiate the ssp with. Can be a single float or a tensor of shape [batch_size]
        length_scale: The width of the kernel

    Returns:
        The fractionally bound vector(s) of shape [num_vsa_dimensions] or [batch_size, num_vsa_dimensions]
    """
    # Handle batch processing if scalar is a tensor
    if isinstance(scalar, torch.Tensor) and scalar.dim() > 0:
        # Ensure ssp is properly shaped for batch processing
        if ssp.dim() == 1:
            # Expand ssp to match batch size
            ssp = ssp.unsqueeze(0).expand(scalar.shape[0], -1)
        
        # Convert to frequency domain
        x = torch.fft.fft(ssp, dim=-1)
        
        # Apply power with broadcasting
        scalar = scalar.unsqueeze(-1) / length_scale
        x = x ** scalar
        
        # Convert back to time domain
        x = torch.fft.ifft(x, dim=-1)
        return x.real
    else:
        # Original single vector processing
        x = torch.fft.fft(ssp)
        x = x ** (scalar / length_scale)
        x = torch.fft.ifft(x)
        return x.real

def make_good_unitary(num_dims: int, device: torch.device, eps: float = 1e-3) -> torch.Tensor:
    """
    Create a hyperdimensional vector of unitary length phasers to build the
    quasi-orthogonal algebraic space.

    Args:
        num_dims: The dimensionality of the VSA
        device: The device to store the tensor on
        eps: The allowable variability in the phase of each phasor

    Returns:
        A one-dimensional tensor of unitary phasors
    """
    a = torch.rand((num_dims - 1) // 2)
    sign = np.random.choice((-1, +1), len(a))
    
    sign = torch.from_numpy(sign).to(device)
    a = a.to(device)

    phi = sign * torch.pi * (eps + a * (1 - 2 * eps))

    assert torch.all(torch.abs(phi) >= torch.pi * eps)
    assert torch.all(torch.abs(phi) <= torch.pi * (1 - eps))

    fv = torch.zeros(num_dims, dtype=torch.complex64, device=device)
    fv[0] = 1
    fv[1:(num_dims + 1) // 2] = torch.cos(phi) + 1j * torch.sin(phi)
    fv[(num_dims // 2) + 1:] = torch.flip(torch.conj(fv[1:(num_dims + 1) // 2]), dims=[0])
    
    if num_dims % 2 == 0:
        fv[num_dims // 2] = 1

    assert torch.allclose(torch.abs(fv), torch.ones(fv.shape, device=device))
    
    v = torch.fft.ifft(fv)
    v = v.real
    v = v.to(device)
    
    assert torch.allclose(torch.fft.fft(v), fv)
    assert torch.allclose(torch.linalg.norm(v), torch.ones(v.shape, device=device))

    return v

class SSPGenerator:
    """
    Spatial Semantic Pointer Generator for creating axis vectors.
    """
    
    def __init__(self, dimensionality: int, device: torch.device, length_scale: float = 1.0):
        """
        Initialize the SSP Generator.
        
        Args:
            dimensionality: The dimensionality of the VSA vectors
            device: The device to store tensors on
            length_scale: The width of the kernel
        """
        self.dimensionality = dimensionality
        self.device = device
        self.length_scale = length_scale
        
    def generate(self, num_axes: int) -> torch.Tensor:
        """
        Generate axis vectors for the specified number of axes.
        
        Args:
            num_axes: The number of axis vectors to generate
            
        Returns:
            A tensor of shape [num_axes, dimensionality] containing the axis vectors
        """
        axis_vectors = torch.zeros((num_axes, self.dimensionality), device=self.device)
        
        for i in range(num_axes):
            axis_vectors[i] = make_good_unitary(self.dimensionality, self.device)
            
        return axis_vectors
