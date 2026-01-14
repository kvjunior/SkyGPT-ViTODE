"""
SkyGPT-ViTODE Neural Network Architectures
===========================================

This module implements the complete neural network architecture for the SkyGPT-ViTODE
framework for probabilistic ultra-short-term solar forecasting.

Architecture Components:
    - Vector Quantized Variational Autoencoder (VQ-VAE) with EMA codebook updates
    - Neural ODE for continuous-time latent dynamics modeling
    - Physics-constrained PhyCell with moment kernel constraints
    - Vision Transformer (ViT) with Mixture Density Network head
    - Conformal prediction for calibrated uncertainty quantification
    - Integrated SkyGPT-ViTODE model with joint training support

Key Features:
    - Configurable architecture parameters via ExperimentConfig
    - Gradient checkpointing for memory-efficient training
    - Distributed training support with synchronized batch normalization
    - Multiple output distribution types (MDN, Gaussian, deterministic)
    - CRPS and NLL loss functions for probabilistic training
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, Dict, List, Any, Union, Callable
from functools import reduce
from dataclasses import dataclass

import numpy as np
from scipy.special import factorial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# Conditional imports for distributed training
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    dist = None

# Conditional import for Neural ODE
try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None
    odeint_adjoint = None


# ============================================================================
# Constants
# ============================================================================

# Numerical stability constants
EPS = 1e-8
LOG_EPS = 1e-10

# Default initialization scale
DEFAULT_INIT_SCALE = 0.02


# ============================================================================
# Activation Function Registry
# ============================================================================

def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        name: Activation function name
        
    Returns:
        Activation module
        
    Raises:
        ValueError: If activation name is not recognized
    """
    activations = {
        "softplus": nn.Softplus(),
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "elu": nn.ELU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "mish": nn.Mish(),
    }
    
    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(activations.keys())}"
        )
    
    return activations[name.lower()]


# ============================================================================
# Utility Functions
# ============================================================================

def shift_dim(
    x: Tensor, 
    src_dim: int = -1, 
    dest_dim: int = -1,
    make_contiguous: bool = True
) -> Tensor:
    """
    Shift a dimension from src_dim to dest_dim.
    
    This is useful for rearranging tensor dimensions without explicit permutation
    specification, particularly when converting between channel-first and
    channel-last formats.
    
    Args:
        x: Input tensor
        src_dim: Source dimension index (supports negative indexing)
        dest_dim: Destination dimension index (supports negative indexing)
        make_contiguous: Whether to make output contiguous in memory
        
    Returns:
        Tensor with shifted dimensions
        
    Example:
        >>> x = torch.randn(2, 3, 4, 5)  # (B, C, H, W)
        >>> y = shift_dim(x, 1, -1)      # (B, H, W, C)
        >>> y.shape
        torch.Size([2, 4, 5, 3])
    """
    n_dims = len(x.shape)
    
    # Handle negative indices
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim
    
    # Validate indices
    if not (0 <= src_dim < n_dims and 0 <= dest_dim < n_dims):
        raise IndexError(
            f"Dimension indices out of range for tensor with {n_dims} dimensions: "
            f"src_dim={src_dim}, dest_dim={dest_dim}"
        )
    
    # Build permutation
    dims = list(range(n_dims))
    dims.pop(src_dim)
    
    permutation = []
    insert_idx = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[insert_idx])
            insert_idx += 1
    
    x = x.permute(permutation)
    
    if make_contiguous:
        x = x.contiguous()
    
    return x


def view_range(
    x: Tensor, 
    start_dim: int, 
    end_dim: Optional[int], 
    shape: Tuple[int, ...]
) -> Tensor:
    """
    Reshape a range of dimensions to a new shape.
    
    Args:
        x: Input tensor
        start_dim: Start dimension (inclusive)
        end_dim: End dimension (exclusive), None for last dimension
        shape: Target shape for the dimension range
        
    Returns:
        Reshaped tensor
    """
    shape = tuple(shape)
    n_dims = len(x.shape)
    
    # Handle negative indices
    if start_dim < 0:
        start_dim = n_dims + start_dim
    if end_dim is None:
        end_dim = n_dims
    elif end_dim < 0:
        end_dim = n_dims + end_dim
    
    # Validate range
    if not (0 <= start_dim < end_dim <= n_dims):
        raise IndexError(
            f"Invalid dimension range [{start_dim}, {end_dim}) "
            f"for tensor with {n_dims} dimensions"
        )
    
    # Build target shape
    x_shape = x.shape
    target_shape = x_shape[:start_dim] + shape + x_shape[end_dim:]
    
    return x.view(target_shape)


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Count model parameters with breakdown by module type.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    # Breakdown by module type
    by_type = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            by_type[module_type] = by_type.get(module_type, 0) + params
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6,
        'by_module_type': by_type
    }


# ============================================================================
# Physics Constraint Utilities (Moment-Kernel Conversion)
# ============================================================================

def tensordot_torch(a: Tensor, b: Tensor, dims: Union[int, Tuple]) -> Tensor:
    """
    PyTorch implementation of numpy's tensordot.
    
    Computes tensor contraction over specified dimensions.
    
    Args:
        a: First input tensor
        b: Second input tensor
        dims: Number of dimensions to contract or tuple of dimension lists
        
    Returns:
        Contracted tensor
    """
    multiply = lambda x, y: x * y
    
    if isinstance(dims, int):
        a = a.contiguous()
        b = b.contiguous()
        
        size_a = a.size()
        size_b = b.size()
        
        size_a0 = size_a[:-dims]
        size_a1 = size_a[-dims:]
        size_b0 = size_b[:dims]
        size_b1 = size_b[dims:]
        
        N = reduce(multiply, size_a1, 1)
        assert reduce(multiply, size_b0, 1) == N, "Contracting dimensions must match"
        
    else:
        a_dims, b_dims = dims
        a_dims = [a_dims] if isinstance(a_dims, int) else list(a_dims)
        b_dims = [b_dims] if isinstance(b_dims, int) else list(b_dims)
        
        # Compute remaining dimensions
        a_remaining = sorted(set(range(a.dim())) - set(a_dims))
        b_remaining = sorted(set(range(b.dim())) - set(b_dims))
        
        # Permute to put contracting dims at end/beginning
        perm_a = a_remaining + a_dims
        perm_b = b_dims + b_remaining
        
        a = a.permute(*perm_a).contiguous()
        b = b.permute(*perm_b).contiguous()
        
        size_a = a.size()
        size_b = b.size()
        
        size_a0 = size_a[:-len(a_dims)]
        size_a1 = size_a[-len(a_dims):]
        size_b0 = size_b[:len(b_dims)]
        size_b1 = size_b[len(b_dims):]
        
        N = reduce(multiply, size_a1, 1)
        assert reduce(multiply, size_b0, 1) == N, "Contracting dimensions must match"
    
    # Reshape for matrix multiplication
    a_flat = a.view(-1, N)
    b_flat = b.view(N, -1)
    
    # Matrix multiply and reshape
    c = a_flat @ b_flat
    return c.view(size_a0 + size_b1)


class MomentKernelBase(nn.Module):
    """
    Base class for moment-kernel conversion matrices.
    
    Implements the mathematical framework for converting between convolution
    kernels and their moment representations, enabling physics-based constraints
    on learned convolution operations.
    
    The moment representation allows enforcing that convolution kernels
    approximate specific spatial derivatives (e.g., Laplacian for diffusion).
    """
    
    def __init__(self, shape: Tuple[int, ...]):
        """
        Initialize moment-kernel conversion matrices.
        
        Args:
            shape: Kernel shape (e.g., (7, 7) for 7x7 kernels)
        """
        super().__init__()
        
        self._size = torch.Size(shape)
        self._dim = len(shape)
        
        if len(shape) == 0:
            raise ValueError("Shape must have at least one dimension")
        
        # Precompute conversion matrices for each dimension
        for j, length in enumerate(shape):
            # Moment matrix M: converts kernel to moments
            M = np.zeros((length, length), dtype=np.float64)
            center = (length - 1) // 2
            
            for i in range(length):
                # M[i, k] = (k - center)^i / i!
                positions = np.arange(length) - center
                M[i] = (positions ** i) / factorial(i)
            
            # Inverse for kernel-to-moment conversion
            try:
                invM = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                invM = np.linalg.pinv(M)
            
            self.register_buffer(f'_M{j}', torch.from_numpy(M).float())
            self.register_buffer(f'_invM{j}', torch.from_numpy(invM).float())
    
    @property
    def M(self) -> List[Tensor]:
        """Get moment matrices for each dimension."""
        return [self._buffers[f'_M{j}'] for j in range(self._dim)]
    
    @property
    def invM(self) -> List[Tensor]:
        """Get inverse moment matrices for each dimension."""
        return [self._buffers[f'_invM{j}'] for j in range(self._dim)]
    
    def size(self) -> torch.Size:
        """Get kernel size."""
        return self._size
    
    def dim(self) -> int:
        """Get number of dimensions."""
        return self._dim
    
    def _pack_dimensions(self, x: Tensor) -> Tensor:
        """Pack batch dimensions for matrix application."""
        if x.dim() < self.dim():
            raise ValueError(
                f"Input must have at least {self.dim()} dimensions, got {x.dim()}"
            )
        
        if x.dim() == self.dim():
            x = x.unsqueeze(0)
        
        x = x.contiguous()
        # Flatten all batch dimensions
        x = x.view(-1, *x.size()[-self.dim():])
        return x
    
    def _apply_matrices_left(self, x: Tensor, matrices: List[Tensor]) -> Tensor:
        """Apply conversion matrices from the left along each axis."""
        if x.dim() != len(matrices) + 1:
            raise ValueError("Dimension mismatch between input and matrices")
        
        original_size = x.size()
        n_axes = x.dim() - 1
        
        # Apply matrices in reverse order (innermost first)
        for i in range(n_axes):
            axis = n_axes - i - 1
            matrix = matrices[axis]
            x = tensordot_torch(matrix, x, dims=([1], [n_axes]))
        
        # Restore batch dimension to front
        x = x.permute([n_axes] + list(range(n_axes))).contiguous()
        x = x.view(original_size)
        
        return x


class M2K(MomentKernelBase):
    """
    Convert moment matrix to convolution kernel.
    
    Given a moment representation, produces the corresponding convolution kernel.
    """
    
    def forward(self, moments: Tensor) -> Tensor:
        """
        Convert moments to kernel.
        
        Args:
            moments: Moment representation (..., *kernel_shape)
            
        Returns:
            Convolution kernel with same shape
        """
        original_size = moments.size()
        moments = self._pack_dimensions(moments)
        kernels = self._apply_matrices_left(moments, self.invM)
        return kernels.view(original_size)


class K2M(MomentKernelBase):
    """
    Convert convolution kernel to moment matrix.
    
    Given a convolution kernel, computes its moment representation.
    This is used to enforce physics constraints on learned kernels.
    """
    
    def forward(self, kernels: Tensor) -> Tensor:
        """
        Convert kernel to moments.
        
        Args:
            kernels: Convolution kernels (..., *kernel_shape)
            
        Returns:
            Moment representation with same shape
        """
        original_size = kernels.size()
        kernels = self._pack_dimensions(kernels)
        moments = self._apply_matrices_left(kernels, self.M)
        return moments.view(original_size)


# ============================================================================
# VQ-VAE Components
# ============================================================================

class SamePadConv3d(nn.Module):
    """
    3D convolution with 'same' padding.
    
    Ensures output spatial dimensions match input dimensions divided by stride,
    regardless of kernel size.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1, 
        bias: bool = True
    ):
        super().__init__()
        
        # Normalize to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Compute padding for each dimension
        # For 'same' padding: pad_total = kernel_size - stride (when stride <= kernel_size)
        total_pad = tuple(max(k - s, 0) for k, s in zip(kernel_size, stride))
        
        # Split padding into left/right for each dimension (reversed for F.pad)
        pad_input = []
        for p in reversed(total_pad):
            pad_left = p // 2
            pad_right = p - pad_left
            pad_input.extend([pad_left, pad_right])
        
        self.pad_input = tuple(pad_input)
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, bias=bias
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    """
    3D transposed convolution with 'same' padding.
    
    Ensures output spatial dimensions match input dimensions multiplied by stride.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1, 
        bias: bool = True
    ):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        # For transposed conv, we need output_padding to get exact dimensions
        # padding = kernel_size - 1, output_padding = stride - 1
        self.convt = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, 
            padding=tuple(k - 1 for k in kernel_size),
            output_padding=tuple(s - 1 for s in stride),
            bias=bias
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.convt(x)


class AxialAttention(nn.Module):
    """
    Axial attention for efficient 3D self-attention.
    
    Instead of full 3D attention (O(T*H*W)^2), applies attention along each
    axis independently, reducing complexity to O(T*H*W * max(T, H, W)).
    
    This is particularly effective for video data where temporal and spatial
    correlations can be modeled separately.
    """
    
    def __init__(self, n_hiddens: int, n_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        
        if n_hiddens % n_heads != 0:
            raise ValueError(
                f"n_hiddens ({n_hiddens}) must be divisible by n_heads ({n_heads})"
            )
        
        self.n_hiddens = n_hiddens
        self.n_heads = n_heads
        
        # Multi-head attention for each axis
        self.attn_t = nn.MultiheadAttention(
            n_hiddens, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_h = nn.MultiheadAttention(
            n_hiddens, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_w = nn.MultiheadAttention(
            n_hiddens, n_heads, dropout=dropout, batch_first=True
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply axial attention.
        
        Args:
            x: Input tensor (B, C, T, H, W)
            
        Returns:
            Output tensor with same shape
        """
        B, C, T, H, W = x.shape
        
        # Temporal attention: attend across T for each (h, w) position
        x_t = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
        x_t, _ = self.attn_t(x_t, x_t, x_t, need_weights=False)
        x_t = x_t.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)
        
        # Height attention: attend across H for each (t, w) position
        x_h = x.permute(0, 2, 4, 3, 1).reshape(B * T * W, H, C)
        x_h, _ = self.attn_h(x_h, x_h, x_h, need_weights=False)
        x_h = x_h.reshape(B, T, W, H, C).permute(0, 4, 1, 3, 2)
        
        # Width attention: attend across W for each (t, h) position
        x_w = x.permute(0, 2, 3, 4, 1).reshape(B * T * H, W, C)
        x_w, _ = self.attn_w(x_w, x_w, x_w, need_weights=False)
        x_w = x_w.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        # Combine with residual connections
        return x_t + x_h + x_w


class AttentionResidualBlock(nn.Module):
    """
    Residual block with axial attention for VQ-VAE.
    
    Combines convolution-based feature extraction with axial attention
    for capturing long-range dependencies in video data.
    """
    
    def __init__(self, n_hiddens: int, n_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        
        # Ensure n_hiddens // 2 works for intermediate channels
        mid_channels = max(n_hiddens // 2, 1)
        
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(inplace=True),
            SamePadConv3d(n_hiddens, mid_channels, 3, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            SamePadConv3d(mid_channels, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(inplace=True),
        )
        
        self.attn = AxialAttention(n_hiddens, n_heads, dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply residual block with attention."""
        h = self.block(x)
        
        # Apply axial attention
        h = self.attn(h)
        
        return x + h


class Codebook(nn.Module):
    """
    Vector quantization codebook with Exponential Moving Average (EMA) updates.
    
    Implements the codebook learning from VQ-VAE with several improvements:
    - EMA updates for stable training without codebook loss gradient
    - Dead code resurrection to prevent codebook collapse
    - Configurable initialization strategies
    
    Reference:
        van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
        Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2", NeurIPS 2019
    """
    
    def __init__(
        self, 
        n_codes: int, 
        embedding_dim: int, 
        decay: float = 0.99,
        epsilon: float = 1e-5,
        commitment_cost: float = 0.25,
        init_strategy: str = "uniform",
        restart_threshold: float = 1.0
    ):
        """
        Initialize codebook.
        
        Args:
            n_codes: Number of codebook entries (vocabulary size)
            embedding_dim: Dimension of each codebook entry
            decay: EMA decay rate (0.99 typical)
            epsilon: Small constant for numerical stability
            commitment_cost: Weight for commitment loss (β in paper)
            init_strategy: Initialization strategy ('uniform', 'normal', 'kmeans')
            restart_threshold: Usage threshold for dead code resurrection
        """
        super().__init__()
        
        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost
        self.restart_threshold = restart_threshold
        self.init_strategy = init_strategy
        
        # Codebook embeddings
        if init_strategy == "normal":
            embeddings = torch.randn(n_codes, embedding_dim) * 0.02
        else:  # uniform or kmeans (kmeans initialized from data)
            embeddings = torch.randn(n_codes, embedding_dim)
        
        self.register_buffer('embeddings', embeddings)
        
        # EMA tracking buffers
        self.register_buffer('N', torch.zeros(n_codes))  # Usage count
        self.register_buffer('z_avg', embeddings.clone())  # Running mean
        
        # Flag for data-driven initialization
        self._need_init = (init_strategy == "kmeans")
    
    def _tile_to_codebook_size(self, x: Tensor) -> Tensor:
        """Tile input vectors to ensure we have enough for initialization."""
        n_vectors, dim = x.shape
        
        if n_vectors < self.n_codes:
            n_repeats = (self.n_codes + n_vectors - 1) // n_vectors
            std = 0.01 / math.sqrt(dim)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        
        return x
    
    def _init_from_data(self, z: Tensor):
        """Initialize codebook from input data (k-means style)."""
        self._need_init = False
        
        # Flatten to (N, embedding_dim)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        
        # Tile if needed and sample random subset
        candidates = self._tile_to_codebook_size(flat_inputs)
        indices = torch.randperm(candidates.shape[0], device=candidates.device)
        selected = candidates[indices[:self.n_codes]]
        
        # Synchronize in distributed setting
        if DISTRIBUTED_AVAILABLE and dist.is_initialized():
            dist.broadcast(selected, src=0)
        
        # Initialize buffers
        self.embeddings.data.copy_(selected)
        self.z_avg.data.copy_(selected)
        self.N.data.fill_(1.0)
    
    def _resurrect_dead_codes(self, flat_inputs: Tensor):
        """Replace underutilized codes with random input vectors."""
        # Compute usage relative to average
        avg_usage = self.N.sum() / self.n_codes
        dead_mask = self.N < (self.restart_threshold * avg_usage)
        n_dead = dead_mask.sum().item()
        
        if n_dead == 0:
            return
        
        # Sample replacement vectors
        candidates = self._tile_to_codebook_size(flat_inputs)
        indices = torch.randperm(candidates.shape[0], device=candidates.device)
        replacements = candidates[indices[:n_dead]]
        
        # Synchronize in distributed setting
        if DISTRIBUTED_AVAILABLE and dist.is_initialized():
            dist.broadcast(replacements, src=0)
        
        # Replace dead codes
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        self.embeddings.data[dead_indices] = replacements
        self.z_avg.data[dead_indices] = replacements
        self.N.data[dead_indices] = avg_usage
    
    def forward(self, z: Tensor) -> Dict[str, Tensor]:
        """
        Quantize input to nearest codebook entries.
        
        Args:
            z: Input tensor (B, embedding_dim, *spatial_dims)
            
        Returns:
            Dictionary containing:
                - embeddings: Quantized embeddings with straight-through gradient
                - encodings: Codebook indices
                - commitment_loss: Commitment loss term
                - perplexity: Codebook utilization metric
        """
        # Initialize from data if needed
        if self._need_init and self.training:
            self._init_from_data(z)
        
        # Flatten spatial dimensions: (B, C, ...) -> (N, C)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        
        # Compute distances to all codebook entries
        # ||z - e||^2 = ||z||^2 - 2*z·e + ||e||^2
        distances = (
            (flat_inputs ** 2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings ** 2).sum(dim=1, keepdim=True).t()
        )
        
        # Find nearest codebook entries
        encoding_indices = distances.argmin(dim=1)
        encodings_onehot = F.one_hot(encoding_indices, self.n_codes).float()
        
        # Reshape indices to spatial dimensions
        spatial_shape = z.shape[2:]
        encoding_indices = encoding_indices.view(z.shape[0], *spatial_shape)
        
        # Look up embeddings
        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)
        
        # Commitment loss: encourage encoder to commit to codebook entries
        commitment_loss = self.commitment_cost * F.mse_loss(z, embeddings.detach())
        
        # EMA codebook update (training only)
        if self.training:
            # Aggregate usage counts
            n_total = encodings_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encodings_onehot
            
            # Synchronize across processes
            if DISTRIBUTED_AVAILABLE and dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)
            
            # EMA updates
            self.N.data.mul_(self.decay).add_(n_total, alpha=1 - self.decay)
            self.z_avg.data.mul_(self.decay).add_(encode_sum.t(), alpha=1 - self.decay)
            
            # Laplace smoothing for stable division
            n_total_smoothed = self.N + self.epsilon
            n_sum = n_total_smoothed.sum()
            weights = n_total_smoothed / n_sum * n_sum
            
            # Update embeddings
            self.embeddings.data.copy_(self.z_avg / weights.unsqueeze(1))
            
            # Resurrect dead codes
            self._resurrect_dead_codes(flat_inputs)
        
        # Straight-through gradient estimator
        embeddings_st = z + (embeddings - z).detach()
        
        # Compute perplexity (effective codebook utilization)
        avg_probs = encodings_onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + LOG_EPS)))
        
        return {
            'embeddings': embeddings_st,
            'encodings': encoding_indices,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'encoding_indices_flat': encoding_indices.flatten()
        }
    
    def lookup(self, encodings: Tensor) -> Tensor:
        """Look up embeddings for given indices."""
        return F.embedding(encodings, self.embeddings)


class VQVAEEncoder(nn.Module):
    """
    VQ-VAE Encoder network.
    
    Progressively downsamples input video through strided convolutions
    and residual attention blocks.
    """
    
    def __init__(
        self, 
        in_channels: int,
        n_hiddens: int, 
        n_res_layers: int,
        downsample: Tuple[int, int, int],
        n_heads: int = 2
    ):
        super().__init__()
        
        # Compute number of downsampling stages per dimension
        n_times_downsample = [int(math.log2(d)) for d in downsample]
        max_ds = max(n_times_downsample)
        
        self.convs = nn.ModuleList()
        
        for i in range(max_ds):
            # First conv takes input channels, rest use n_hiddens
            in_ch = in_channels if i == 0 else n_hiddens
            
            # Stride is 2 for dimensions that still need downsampling
            stride = tuple(2 if n_times_downsample[d] > i else 1 for d in range(3))
            
            conv = SamePadConv3d(in_ch, n_hiddens, kernel_size=4, stride=stride)
            self.convs.append(conv)
        
        # Final 3x3 conv
        self.conv_last = SamePadConv3d(n_hiddens, n_hiddens, kernel_size=3)
        
        # Residual stack with attention
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens, n_heads) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Encode input video to latent representation."""
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return h


class VQVAEDecoder(nn.Module):
    """
    VQ-VAE Decoder network.
    
    Progressively upsamples latent representation through transposed
    convolutions and residual attention blocks.
    """
    
    def __init__(
        self, 
        out_channels: int,
        n_hiddens: int, 
        n_res_layers: int,
        upsample: Tuple[int, int, int],
        n_heads: int = 2
    ):
        super().__init__()
        
        # Residual stack first
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens, n_heads) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(inplace=True)
        )
        
        # Compute upsampling stages
        n_times_upsample = [int(math.log2(u)) for u in upsample]
        max_us = max(n_times_upsample)
        
        self.convts = nn.ModuleList()
        
        for i in range(max_us):
            # Last conv outputs target channels, rest use n_hiddens
            out_ch = out_channels if i == max_us - 1 else n_hiddens
            
            # Stride is 2 for dimensions that still need upsampling
            remaining = [n_times_upsample[d] - i for d in range(3)]
            stride = tuple(2 if remaining[d] > 0 else 1 for d in range(3))
            
            convt = SamePadConvTranspose3d(n_hiddens, out_ch, kernel_size=4, stride=stride)
            self.convts.append(convt)
    
    def forward(self, x: Tensor) -> Tensor:
        """Decode latent representation to video."""
        h = self.res_stack(x)
        
        for i, convt in enumerate(self.convts):
            h = convt(h)
            # Apply ReLU except on final layer
            if i < len(self.convts) - 1:
                h = F.relu(h)
        
        return h


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder for video compression.
    
    Learns discrete latent representations of sky video sequences through:
    - Hierarchical encoding with strided 3D convolutions
    - Vector quantization with EMA codebook learning
    - Hierarchical decoding with transposed 3D convolutions
    
    The discrete latent space enables efficient autoregressive modeling
    of video sequences using transformer architectures.
    
    Reference:
        van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
    """
    
    def __init__(
        self, 
        embedding_dim: int = 256, 
        n_codes: int = 2048,
        n_hiddens: int = 240, 
        n_res_layers: int = 4,
        downsample: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 3,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        codebook_init: str = "uniform",
        restart_threshold: float = 1.0,
        sequence_length: int = 16, 
        resolution: int = 64
    ):
        """
        Initialize VQ-VAE.
        
        Args:
            embedding_dim: Dimension of codebook embeddings
            n_codes: Number of codebook entries
            n_hiddens: Number of hidden channels in encoder/decoder
            n_res_layers: Number of residual blocks
            downsample: Downsampling factors for (T, H, W)
            in_channels: Number of input channels (3 for RGB)
            commitment_cost: Weight for commitment loss
            decay: EMA decay rate for codebook
            epsilon: Numerical stability constant
            codebook_init: Codebook initialization strategy
            restart_threshold: Threshold for dead code resurrection
            sequence_length: Input sequence length (for shape computation)
            resolution: Input spatial resolution (for shape computation)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_codes = n_codes
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.downsample = downsample
        
        # Encoder
        self.encoder = VQVAEEncoder(
            in_channels=in_channels,
            n_hiddens=n_hiddens,
            n_res_layers=n_res_layers,
            downsample=downsample
        )
        
        # Pre-quantization projection
        self.pre_vq_conv = SamePadConv3d(n_hiddens, embedding_dim, kernel_size=1)
        
        # Codebook
        self.codebook = Codebook(
            n_codes=n_codes,
            embedding_dim=embedding_dim,
            decay=decay,
            epsilon=epsilon,
            commitment_cost=commitment_cost,
            init_strategy=codebook_init,
            restart_threshold=restart_threshold
        )
        
        # Post-quantization projection
        self.post_vq_conv = SamePadConv3d(embedding_dim, n_hiddens, kernel_size=1)
        
        # Decoder
        self.decoder = VQVAEDecoder(
            out_channels=in_channels,
            n_hiddens=n_hiddens,
            n_res_layers=n_res_layers,
            upsample=downsample
        )
    
    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        """Compute latent spatial dimensions."""
        return tuple(
            s // d for s, d in zip(
                (self.sequence_length, self.resolution, self.resolution),
                self.downsample
            )
        )
    
    def encode(
        self, 
        x: Tensor, 
        include_embeddings: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Encode input video to discrete latent codes.
        
        Args:
            x: Input video (B, C, T, H, W)
            include_embeddings: Whether to return continuous embeddings too
            
        Returns:
            Encoding indices, optionally with continuous embeddings
        """
        h = self.encoder(x)
        h = self.pre_vq_conv(h)
        vq_output = self.codebook(h)
        
        if include_embeddings:
            return vq_output['encodings'], vq_output['embeddings']
        return vq_output['encodings']
    
    def decode(self, encodings: Tensor) -> Tensor:
        """
        Decode discrete codes to video.
        
        Args:
            encodings: Codebook indices (B, T', H', W')
            
        Returns:
            Reconstructed video (B, C, T, H, W)
        """
        h = self.codebook.lookup(encodings)
        h = shift_dim(h, -1, 1)
        h = self.post_vq_conv(h)
        return self.decoder(h)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Full forward pass: encode, quantize, decode.
        
        Args:
            x: Input video (B, C, T, H, W)
            
        Returns:
            Tuple of (reconstruction_loss, reconstructed_video, vq_output_dict)
        """
        # Encode
        h = self.encoder(x)
        h = self.pre_vq_conv(h)
        
        # Quantize
        vq_output = self.codebook(h)
        
        # Decode
        h_q = self.post_vq_conv(vq_output['embeddings'])
        x_recon = self.decoder(h_q)
        
        # Reconstruction loss (normalized by data variance ~0.06 for images in [-0.5, 0.5])
        recon_loss = F.mse_loss(x_recon, x) / 0.06
        
        return recon_loss, x_recon, vq_output


# ============================================================================
# Neural ODE Components
# ============================================================================

class ODEFunc(nn.Module):
    """
    Neural network defining ODE dynamics: dh/dt = f(h, t).
    
    Implements continuous-time dynamics for latent state evolution.
    The network takes the current state and time as input and outputs
    the instantaneous rate of change.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        n_layers: int = 3,
        activation: str = "softplus",
        time_embedding_dim: int = 16
    ):
        """
        Initialize ODE function network.
        
        Args:
            hidden_dim: Dimension of hidden state
            n_layers: Number of MLP layers
            activation: Activation function name
            time_embedding_dim: Dimension of time embedding
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Main dynamics network
        act_fn = get_activation(activation)
        
        layers = []
        for i in range(n_layers):
            in_dim = hidden_dim + time_embedding_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            
            if i < n_layers - 1:
                layers.append(act_fn)
        
        self.net = nn.Sequential(*layers)
        
        # Initialize output layer to near-zero for stable training
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)
        
        # Track number of function evaluations
        self.nfe = 0
    
    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        """
        Compute dynamics at time t.
        
        Args:
            t: Current time (scalar tensor)
            h: Current hidden state (B, hidden_dim) or (B, hidden_dim, H, W)
            
        Returns:
            Rate of change dh/dt with same shape as h
        """
        self.nfe += 1
        
        # Store original shape for reshaping output
        original_shape = h.shape
        batch_size = h.shape[0]
        
        # Flatten spatial dimensions if present
        if h.dim() > 2:
            h_flat = h.view(batch_size, -1)
        else:
            h_flat = h
        
        # Time embedding
        t_emb = self.time_embed(t.view(1, 1)).expand(batch_size, -1)
        
        # Concatenate state and time
        h_t = torch.cat([h_flat, t_emb], dim=-1)
        
        # Compute dynamics
        out = self.net(h_t)
        
        # Reshape to original spatial dimensions
        if len(original_shape) > 2:
            out = out.view(original_shape)
        
        return out


class NeuralODEDynamics(nn.Module):
    """
    Neural ODE module for continuous-time latent dynamics.
    
    Models the evolution of latent representations as a continuous-time
    dynamical system, enabling smooth interpolation between frames and
    physics-informed temporal evolution.
    
    Features:
    - Adaptive ODE solvers for accuracy/speed tradeoff
    - Adjoint method for memory-efficient backpropagation
    - Kinetic energy regularization for smooth trajectories
    
    Reference:
        Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        n_layers: int = 3,
        solver: str = "dopri5", 
        rtol: float = 1e-3,
        atol: float = 1e-4, 
        adjoint: bool = True,
        activation: str = "softplus",
        max_num_steps: int = 1000
    ):
        """
        Initialize Neural ODE.
        
        Args:
            hidden_dim: Dimension of hidden state
            n_layers: Number of layers in ODE function
            solver: ODE solver algorithm
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
            adjoint: Whether to use adjoint method for backprop
            activation: Activation function
            max_num_steps: Maximum solver steps (prevents infinite loops)
        """
        super().__init__()
        
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError(
                "torchdiffeq is required for Neural ODE. "
                "Install via: pip install torchdiffeq"
            )
        
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.max_num_steps = max_num_steps
        
        # ODE function network
        self.odefunc = ODEFunc(hidden_dim, n_layers, activation)
        
        # Select integration method
        self._odeint = odeint_adjoint if adjoint else odeint
    
    def forward(
        self, 
        h0: Tensor, 
        t_span: Tensor,
        return_trajectory: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Integrate ODE from initial condition over time span.
        
        Args:
            h0: Initial hidden state (B, hidden_dim) or (B, hidden_dim, H, W)
            t_span: Time points for evaluation (T,)
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Hidden state at final time, or (trajectory, final_state) if return_trajectory
        """
        self.odefunc.nfe = 0
        
        # Ensure t_span is on same device
        t_span = t_span.to(h0.device)
        
        # Integrate ODE
        options = {'max_num_steps': self.max_num_steps}
        
        h_trajectory = self._odeint(
            self.odefunc, 
            h0, 
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            options=options
        )
        
        if return_trajectory:
            return h_trajectory, h_trajectory[-1]
        return h_trajectory[-1]
    
    @property
    def nfe(self) -> int:
        """Number of function evaluations in last forward pass."""
        return self.odefunc.nfe
    
    def compute_kinetic_energy(
        self, 
        h_trajectory: Tensor, 
        t_span: Tensor
    ) -> Tensor:
        """
        Compute kinetic energy regularization term.
        
        Penalizes rapid changes in trajectory to encourage smooth dynamics.
        
        Args:
            h_trajectory: Trajectory tensor (T, B, ...)
            t_span: Time points (T,)
            
        Returns:
            Kinetic energy scalar
        """
        if len(t_span) < 2:
            return torch.tensor(0.0, device=h_trajectory.device)
        
        # Compute time differences
        dt = t_span[1:] - t_span[:-1]
        
        # Compute state differences
        dh = h_trajectory[1:] - h_trajectory[:-1]
        
        # Velocity = dh/dt
        # Need to broadcast dt to match dh shape
        dt_shape = [len(dt)] + [1] * (dh.dim() - 1)
        velocity = dh / dt.view(*dt_shape)
        
        # Kinetic energy = mean squared velocity
        kinetic = (velocity ** 2).mean()
        
        return kinetic


# ============================================================================
# Physics-Constrained PhyCell
# ============================================================================

class PhyCellUnit(nn.Module):
    """
    Single physics-constrained recurrent cell.
    
    Implements a prediction-correction scheme:
    - Prediction: h_tilde = h + F(h) approximates PDE-governed dynamics
    - Correction: h_next = h_tilde + K * (x - h_tilde) incorporates observations
    
    The F network learns to approximate spatial derivatives, constrained
    by moment conditions to ensure physical consistency.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
        kernel_size: Tuple[int, int] = (7, 7)
    ):
        """
        Initialize PhyCell unit.
        
        Args:
            input_dim: Number of input/output channels
            hidden_dim: Hidden dimension for dynamics network
            kernel_size: Spatial kernel size for convolutions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # Compute valid number of groups for GroupNorm
        # Find largest divisor of hidden_dim that is <= sqrt(hidden_dim)
        n_groups = 1
        sqrt_hidden = int(math.sqrt(hidden_dim))
        for g in range(sqrt_hidden, 0, -1):
            if hidden_dim % g == 0:
                n_groups = g
                break
        
        # Physics dynamics approximation F
        self.F = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        )
        
        # Correction gate
        self.gate = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        """
        Process one timestep.
        
        Args:
            x: Input observation (B, C, H, W)
            hidden: Previous hidden state (B, C, H, W)
            
        Returns:
            Next hidden state (B, C, H, W)
        """
        # Prediction step: h_tilde = h + F(h)
        hidden_tilde = hidden + self.F(hidden)
        
        # Correction step with learned gate
        combined = torch.cat([x, hidden], dim=1)
        K = self.gate(combined)
        
        # h_next = h_tilde + K * (x - h_tilde)
        next_hidden = hidden_tilde + K * (x - hidden_tilde)
        
        return next_hidden


class PhyCell(nn.Module):
    """
    Physics-constrained cell module for cloud dynamics modeling.
    
    Stacks multiple PhyCellUnits with moment constraints to ensure
    convolution kernels approximate physical spatial derivatives
    (e.g., Laplacian for diffusion, gradients for advection).
    
    Reference:
        Le Guen & Thome, "Disentangling Physical Dynamics from Unknown Factors
        for Unsupervised Video Prediction", CVPR 2020
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        n_layers: int = 1,
        kernel_size: int = 7, 
        use_physics_constraint: bool = True
    ):
        """
        Initialize PhyCell.
        
        Args:
            input_dim: Number of input channels
            hidden_dim: Hidden dimension for cell units
            n_layers: Number of stacked PhyCellUnits
            kernel_size: Spatial kernel size
            use_physics_constraint: Whether to apply moment constraints
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.use_physics_constraint = use_physics_constraint
        
        # Stack of PhyCell units
        self.cells = nn.ModuleList([
            PhyCellUnit(input_dim, hidden_dim, (kernel_size, kernel_size))
            for _ in range(n_layers)
        ])
        
        # Physics constraint matrices
        if use_physics_constraint:
            self.k2m = K2M([kernel_size, kernel_size])
            
            # Target moment constraints (identity at each position)
            # This encourages kernels to compute spatial derivatives
            constraints = torch.zeros(kernel_size ** 2, kernel_size, kernel_size)
            for i in range(kernel_size):
                for j in range(kernel_size):
                    constraints[i * kernel_size + j, i, j] = 1.0
            self.register_buffer('moment_constraints', constraints)
        
        # Hidden state storage
        self._hidden_states: Optional[List[Tensor]] = None
    
    def init_hidden(self, x: Tensor) -> List[Tensor]:
        """Initialize hidden states to zeros matching input shape."""
        return [torch.zeros_like(x) for _ in range(self.n_layers)]
    
    def forward(
        self, 
        x: Tensor, 
        hidden_states: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Process one timestep through all layers.
        
        Args:
            x: Input tensor (B, C, H, W)
            hidden_states: List of hidden states for each layer
            
        Returns:
            Tuple of (output, new_hidden_states)
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(x)
        
        new_hidden_states = []
        
        for i, cell in enumerate(self.cells):
            inp = x if i == 0 else new_hidden_states[i - 1]
            new_h = cell(inp, hidden_states[i])
            new_hidden_states.append(new_h)
        
        return new_hidden_states[-1], new_hidden_states
    
    def forward_sequence(
        self, 
        x_seq: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Process a sequence of inputs.
        
        Args:
            x_seq: Input sequence (B, T, C, H, W)
            
        Returns:
            Tuple of (output_sequence, physics_loss)
        """
        B, T, C, H, W = x_seq.shape
        
        hidden_states = None
        outputs = []
        
        for t in range(T):
            out, hidden_states = self(x_seq[:, t], hidden_states)
            outputs.append(out)
        
        output_seq = torch.stack(outputs, dim=1)
        physics_loss = self.compute_physics_loss()
        
        return output_seq, physics_loss
    
    def compute_physics_loss(self) -> Tensor:
        """
        Compute physics constraint loss.
        
        Penalizes deviation of learned kernel moments from target constraints,
        encouraging physically meaningful spatial operations.
        
        Returns:
            Physics constraint loss scalar
        """
        if not self.use_physics_constraint:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for cell in self.cells:
            # Get first conv layer weights: (out_ch, in_ch, kH, kW)
            weights = cell.F[0].weight
            out_ch, in_ch, kH, kW = weights.shape
            
            # Process each input channel
            for c in range(min(in_ch, self.input_dim)):
                # Extract filters for this input channel
                filters = weights[:, c, :, :]  # (out_ch, kH, kW)
                
                # Convert to moments
                moments = self.k2m(filters)
                
                # Compare to constraints (use subset if out_ch < k^2)
                n_compare = min(out_ch, self.moment_constraints.shape[0])
                target = self.moment_constraints[:n_compare].to(moments.device)
                
                loss = loss + F.mse_loss(moments[:n_compare], target)
        
        return loss


# ============================================================================
# Vision Transformer Components
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert image to sequence of patch embeddings.
    
    Splits image into non-overlapping patches and projects each
    to the embedding dimension.
    """
    
    def __init__(
        self, 
        img_size: int = 64, 
        patch_size: int = 8,
        in_channels: int = 3, 
        embed_dim: int = 384
    ):
        """
        Initialize patch embedding.
        
        Args:
            img_size: Input image size (assumes square)
            patch_size: Size of each patch (assumes square)
            in_channels: Number of input channels
            embed_dim: Output embedding dimension
        """
        super().__init__()
        
        if img_size % patch_size != 0:
            raise ValueError(
                f"Image size ({img_size}) must be divisible by patch size ({patch_size})"
            )
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Linear projection via convolution
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Convert image to patch embeddings.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            Patch embeddings (B, n_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Validate input size
        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"Input size ({H}x{W}) doesn't match expected ({self.img_size}x{self.img_size})"
            )
        
        # Project patches: (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        
        # Flatten spatial dims: (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention layer.
    
    Implements scaled dot-product attention with multiple heads
    for capturing different types of relationships.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int = 8,
        qkv_bias: bool = True, 
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Input/output embedding dimension
            n_heads: Number of attention heads
            qkv_bias: Whether to include bias in QKV projections
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for output projection
        """
        super().__init__()
        
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
            )
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self, 
        x: Tensor, 
        return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor (B, N, C)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor, optionally with attention weights
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    Randomly drops entire residual branches during training,
    enabling training of very deep networks.
    
    Reference:
        Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        
        # Sample binary mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = (random_tensor < keep_prob).float()
        
        # Scale by keep probability
        return x * binary_mask / keep_prob
    
    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class MLP(nn.Module):
    """MLP block for transformer."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    
    Implements pre-norm architecture with residual connections
    and optional stochastic depth.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int = 8,
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = True,
        drop: float = 0.0, 
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, n_heads, qkv_bias, attn_drop, drop
        )
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden, drop=drop)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply transformer block."""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TemporalCrossAttention(nn.Module):
    """
    Cross-attention for temporal context integration.
    
    Allows current frame features to attend to historical
    frame features for temporal reasoning.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int = 8,
        qkv_bias: bool = True, 
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
            )
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate projections for query and key/value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        """
        Apply cross-attention.
        
        Args:
            query: Query tensor (B, N, C)
            context: Context tensor (B, M, C)
            
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = query.shape
        M = context.shape[1]
        
        # Project query
        q = self.q_proj(query).reshape(B, N, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, heads, N, head_dim)
        
        # Project key and value from context
        kv = self.kv_proj(context).reshape(B, M, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, heads, M, head_dim)
        k, v = kv.unbind(0)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# ============================================================================
# Mixture Density Network
# ============================================================================

class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network output head for probabilistic predictions.
    
    Models the conditional distribution p(y|x) as a mixture of Gaussians,
    enabling multi-modal uncertainty representation.
    
    Reference:
        Bishop, "Mixture Density Networks", 1994
    """
    
    def __init__(
        self, 
        input_dim: int, 
        n_components: int = 5,
        eps: float = 1e-6
    ):
        """
        Initialize MDN.
        
        Args:
            input_dim: Input feature dimension
            n_components: Number of Gaussian components
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        self.n_components = n_components
        self.eps = eps
        
        # Output projections for mixture parameters
        self.fc_weights = nn.Linear(input_dim, n_components)
        self.fc_means = nn.Linear(input_dim, n_components)
        self.fc_log_scales = nn.Linear(input_dim, n_components)
        
        # Initialize to reasonable defaults
        nn.init.zeros_(self.fc_weights.bias)
        nn.init.zeros_(self.fc_means.bias)
        nn.init.constant_(self.fc_log_scales.bias, -1.0)  # Start with small variance
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Compute mixture parameters.
        
        Args:
            x: Input features (B, input_dim)
            
        Returns:
            Dictionary with mixture parameters
        """
        # Mixture weights (sum to 1)
        weights = F.softmax(self.fc_weights(x), dim=-1)
        
        # Component means
        means = self.fc_means(x)
        
        # Component scales (positive via exp)
        log_scales = self.fc_log_scales(x)
        scales = torch.exp(log_scales).clamp(min=self.eps)
        
        return {
            'weights': weights,      # (B, K)
            'means': means,          # (B, K)
            'scales': scales,        # (B, K)
            'log_scales': log_scales # (B, K)
        }
    
    def sample(
        self, 
        params: Dict[str, Tensor], 
        n_samples: int = 1,
        temperature: float = 1.0
    ) -> Tensor:
        """
        Sample from mixture distribution.
        
        Args:
            params: Mixture parameters from forward()
            n_samples: Number of samples per input
            temperature: Sampling temperature (1.0 = standard)
            
        Returns:
            Samples (B, n_samples)
        """
        weights = params['weights']
        means = params['means']
        scales = params['scales'] * temperature
        
        B, K = weights.shape
        
        # Sample component indices
        component_dist = torch.distributions.Categorical(weights)
        component_idx = component_dist.sample((n_samples,))  # (n_samples, B)
        component_idx = component_idx.t()  # (B, n_samples)
        
        # Sample from selected Gaussians
        samples = []
        for s in range(n_samples):
            idx = component_idx[:, s:s+1]  # (B, 1)
            
            mean = torch.gather(means, 1, idx).squeeze(1)
            scale = torch.gather(scales, 1, idx).squeeze(1)
            
            sample = mean + scale * torch.randn_like(mean)
            samples.append(sample)
        
        return torch.stack(samples, dim=1)  # (B, n_samples)
    
    def get_mean(self, params: Dict[str, Tensor]) -> Tensor:
        """Get expected value of mixture."""
        return (params['weights'] * params['means']).sum(dim=-1)
    
    def get_mode(self, params: Dict[str, Tensor]) -> Tensor:
        """Get mode (mean of most likely component)."""
        idx = params['weights'].argmax(dim=-1, keepdim=True)
        return torch.gather(params['means'], 1, idx).squeeze(1)
    
    def get_variance(self, params: Dict[str, Tensor]) -> Tensor:
        """Get variance of mixture (law of total variance)."""
        weights = params['weights']
        means = params['means']
        scales = params['scales']
        
        # Mean of mixture
        mixture_mean = self.get_mean(params).unsqueeze(-1)
        
        # Variance within components: E[Var(Y|Z)]
        var_within = (weights * scales ** 2).sum(dim=-1)
        
        # Variance between components: Var(E[Y|Z])
        var_between = (weights * (means - mixture_mean) ** 2).sum(dim=-1)
        
        return var_within + var_between
    
    def get_std(self, params: Dict[str, Tensor]) -> Tensor:
        """Get standard deviation of mixture."""
        return torch.sqrt(self.get_variance(params) + self.eps)
    
    def nll_loss(
        self, 
        params: Dict[str, Tensor], 
        target: Tensor,
        reduction: str = 'mean'
    ) -> Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            params: Mixture parameters
            target: Target values (B,)
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            NLL loss
        """
        weights = params['weights']      # (B, K)
        means = params['means']          # (B, K)
        log_scales = params['log_scales'] # (B, K)
        
        target = target.unsqueeze(-1)    # (B, 1)
        
        # Log probability for each component
        # log N(y; μ, σ) = -0.5 * log(2π) - log(σ) - 0.5 * ((y-μ)/σ)²
        log_probs = (
            -0.5 * math.log(2 * math.pi)
            - log_scales
            - 0.5 * ((target - means) / (torch.exp(log_scales) + self.eps)) ** 2
        )
        
        # Log mixture probability via log-sum-exp
        log_weights = torch.log(weights + LOG_EPS)
        log_mixture = torch.logsumexp(log_weights + log_probs, dim=-1)
        
        # Negative log-likelihood
        nll = -log_mixture
        
        if reduction == 'mean':
            return nll.mean()
        elif reduction == 'sum':
            return nll.sum()
        return nll
    
    def crps_loss(
        self, 
        params: Dict[str, Tensor], 
        target: Tensor,
        n_samples: int = 100,
        reduction: str = 'mean'
    ) -> Tensor:
        """
        Compute Continuous Ranked Probability Score (CRPS).
        
        CRPS is a proper scoring rule for probabilistic forecasts.
        Computed via Monte Carlo estimation.
        
        Args:
            params: Mixture parameters
            target: Target values (B,)
            n_samples: Number of MC samples
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            CRPS loss
        """
        # Sample from mixture
        samples = self.sample(params, n_samples)  # (B, n_samples)
        
        # CRPS = E|Y - y| - 0.5 * E|Y - Y'|
        # where Y, Y' are independent samples from predictive distribution
        
        target_expanded = target.unsqueeze(-1)  # (B, 1)
        
        # E|Y - y|
        term1 = torch.abs(samples - target_expanded).mean(dim=-1)
        
        # E|Y - Y'|: compute pairwise differences
        samples_diff = samples.unsqueeze(-1) - samples.unsqueeze(-2)
        term2 = torch.abs(samples_diff).mean(dim=(-1, -2))
        
        crps = term1 - 0.5 * term2
        
        if reduction == 'mean':
            return crps.mean()
        elif reduction == 'sum':
            return crps.sum()
        return crps


# ============================================================================
# Vision Transformer PV Predictor
# ============================================================================

class VisionTransformerPVPredictor(nn.Module):
    """
    Vision Transformer for PV power prediction from sky images.
    
    Processes sky images through patch embedding and transformer blocks,
    with optional temporal cross-attention for integrating historical context.
    Outputs probabilistic predictions via configurable distribution heads.
    """
    
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_temporal_attention: bool = True,
        temporal_depth: int = 2,
        output_type: str = "mdn",
        n_mixture_components: int = 5,
        init_scale: float = 0.02
    ):
        """
        Initialize ViT PV predictor.
        
        Args:
            img_size: Input image size
            patch_size: Patch size for embedding
            in_channels: Number of input channels
            embed_dim: Transformer embedding dimension
            depth: Number of transformer blocks
            n_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Include bias in QKV projections
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            use_temporal_attention: Whether to use temporal cross-attention
            temporal_depth: Number of temporal attention blocks
            output_type: Output distribution type ('mdn', 'gaussian', 'deterministic')
            n_mixture_components: Number of MDN components
            init_scale: Weight initialization scale
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.output_type = output_type
        self.use_temporal_attention = use_temporal_attention
        self.init_scale = init_scale
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable tokens and embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth schedule (linear increase)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, n_heads, mlp_ratio, qkv_bias,
                drop_rate, attn_drop_rate, dpr[i]
            )
            for i in range(depth)
        ])
        
        # Temporal cross-attention (optional)
        if use_temporal_attention:
            self.temporal_norm = nn.LayerNorm(embed_dim)
            self.temporal_blocks = nn.ModuleList([
                TemporalCrossAttention(
                    embed_dim, n_heads, qkv_bias, attn_drop_rate, drop_rate
                )
                for _ in range(temporal_depth)
            ])
            self.temporal_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head
        if output_type == "mdn":
            self.head = MixtureDensityNetwork(embed_dim, n_mixture_components)
        elif output_type == "gaussian":
            self.head = nn.Linear(embed_dim, 2)  # mean and log_var
        else:  # deterministic
            self.head = nn.Linear(embed_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Position and class token
        nn.init.trunc_normal_(self.pos_embed, std=self.init_scale)
        nn.init.trunc_normal_(self.cls_token, std=self.init_scale)
        
        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=self.init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: Tensor,
        temporal_context: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass for PV prediction.
        
        Args:
            x: Input sky image (B, C, H, W)
            temporal_context: Historical frame embeddings (B, T, embed_dim)
            
        Returns:
            Dictionary with prediction outputs
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)
        
        # Add position embedding
        x = self.pos_drop(x + self.pos_embed)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Temporal cross-attention
        if self.use_temporal_attention and temporal_context is not None:
            context = self.temporal_proj(temporal_context)
            for temporal_block in self.temporal_blocks:
                x_norm = self.temporal_norm(x)
                x = x + temporal_block(x_norm, context)
        
        # Final normalization
        x = self.norm(x)
        
        # Use class token for prediction
        cls_output = x[:, 0]  # (B, embed_dim)
        
        # Output head
        if self.output_type == "mdn":
            output = self.head(cls_output)
            output['mean'] = self.head.get_mean(output)
            output['std'] = self.head.get_std(output)
        elif self.output_type == "gaussian":
            out = self.head(cls_output)
            mean = out[:, 0]
            log_var = out[:, 1]
            output = {
                'mean': mean,
                'log_var': log_var,
                'std': torch.exp(0.5 * log_var),
                'weights': None,
                'means': mean.unsqueeze(-1),
                'scales': torch.exp(0.5 * log_var).unsqueeze(-1)
            }
        else:  # deterministic
            pred = self.head(cls_output).squeeze(-1)
            output = {
                'mean': pred,
                'prediction': pred,
                'std': torch.zeros_like(pred),
                'weights': None,
                'means': None,
                'scales': None
            }
        
        output['cls_features'] = cls_output
        
        return output
    
    def compute_loss(
        self,
        output: Dict[str, Tensor],
        target: Tensor,
        crps_weight: float = 0.0
    ) -> Dict[str, Tensor]:
        """
        Compute prediction loss.
        
        Args:
            output: Model output dictionary
            target: Target values (B,)
            crps_weight: Weight for CRPS loss component
            
        Returns:
            Dictionary with loss components
        """
        if self.output_type == "mdn":
            nll_loss = self.head.nll_loss(output, target)
            
            losses = {'nll': nll_loss}
            
            if crps_weight > 0:
                crps_loss = self.head.crps_loss(output, target)
                losses['crps'] = crps_loss
                losses['total'] = nll_loss + crps_weight * crps_loss
            else:
                losses['total'] = nll_loss
                
        elif self.output_type == "gaussian":
            mean = output['mean']
            log_var = output['log_var']
            
            # Gaussian NLL
            nll_loss = 0.5 * (log_var + (target - mean) ** 2 / (torch.exp(log_var) + EPS))
            nll_loss = nll_loss.mean()
            
            losses = {'nll': nll_loss, 'total': nll_loss}
            
        else:  # deterministic
            mse_loss = F.mse_loss(output['prediction'], target)
            losses = {'mse': mse_loss, 'total': mse_loss}
        
        return losses


# ============================================================================
# Conformal Prediction
# ============================================================================

class ConformalPredictor(nn.Module):
    """
    Conformal prediction wrapper for calibrated uncertainty quantification.
    
    Provides distribution-free coverage guarantees for prediction intervals
    through post-hoc calibration on a holdout set.
    
    Reference:
        Vovk et al., "Algorithmic Learning in a Random World", 2005
        Romano et al., "Conformalized Quantile Regression", NeurIPS 2019
    """
    
    def __init__(
        self,
        coverage_level: float = 0.90,
        adaptive: bool = False,
        n_neighbors: int = 100
    ):
        """
        Initialize conformal predictor.
        
        Args:
            coverage_level: Target coverage probability
            adaptive: Whether to use adaptive conformal prediction
            n_neighbors: Number of neighbors for adaptive method
        """
        super().__init__()
        
        if not 0 < coverage_level < 1:
            raise ValueError(f"coverage_level must be in (0, 1), got {coverage_level}")
        
        self.coverage_level = coverage_level
        self.adaptive = adaptive
        self.n_neighbors = n_neighbors
        
        # Calibration state
        self.register_buffer('calibrated_quantile', torch.tensor(float('inf')))
        self.register_buffer('calibration_scores', torch.tensor([]))
        self.is_calibrated = False
    
    def calibrate(
        self,
        predictions: Tensor,
        uncertainties: Tensor,
        targets: Tensor
    ):
        """
        Calibrate using holdout calibration set.
        
        Computes nonconformity scores and determines the quantile
        needed to achieve target coverage.
        
        Args:
            predictions: Point predictions (N,)
            uncertainties: Predicted uncertainties (N,)
            targets: True values (N,)
        """
        # Compute nonconformity scores
        # Score = |y - pred| / uncertainty
        scores = torch.abs(targets - predictions) / (uncertainties + EPS)
        
        # Sort scores
        sorted_scores, _ = torch.sort(scores)
        n = len(sorted_scores)
        
        # Compute quantile level with finite sample correction
        # q = ceil((n+1) * coverage) / n
        q_level = math.ceil((n + 1) * self.coverage_level) / n
        q_level = min(q_level, 1.0)
        
        # Get quantile value
        idx = int(q_level * n) - 1
        idx = max(0, min(idx, n - 1))
        
        self.calibrated_quantile = sorted_scores[idx]
        self.calibration_scores = sorted_scores
        self.is_calibrated = True
    
    def get_intervals(
        self,
        predictions: Tensor,
        uncertainties: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Get calibrated prediction intervals.
        
        Args:
            predictions: Point predictions (N,)
            uncertainties: Predicted uncertainties (N,)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if not self.is_calibrated:
            # Use standard Gaussian quantile if not calibrated
            from scipy import stats
            multiplier = stats.norm.ppf((1 + self.coverage_level) / 2)
        else:
            multiplier = self.calibrated_quantile.item()
        
        margin = multiplier * uncertainties
        
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper
    
    def compute_coverage(
        self,
        predictions: Tensor,
        uncertainties: Tensor,
        targets: Tensor
    ) -> float:
        """Compute empirical coverage on test set."""
        lower, upper = self.get_intervals(predictions, uncertainties)
        covered = (targets >= lower) & (targets <= upper)
        return covered.float().mean().item()
    
    def compute_interval_width(
        self,
        predictions: Tensor,
        uncertainties: Tensor
    ) -> Tensor:
        """Compute prediction interval widths."""
        lower, upper = self.get_intervals(predictions, uncertainties)
        return upper - lower


# ============================================================================
# Integrated SkyGPT-ViTODE Model
# ============================================================================

class SkyGPTViTODE(nn.Module):
    """
    SkyGPT-ViTODE: Integrated probabilistic solar forecasting framework.
    
    Combines multiple components for end-to-end sky image to PV power prediction:
    
    1. VQ-VAE: Discrete latent representation of sky videos
    2. PhyCell: Physics-constrained recurrent prediction
    3. Neural ODE: Continuous-time latent dynamics
    4. Vision Transformer: Image-to-power mapping with MDN output
    5. Conformal Prediction: Calibrated uncertainty quantification
    
    The framework supports:
    - Joint training of video prediction and PV forecasting
    - Multiple output distribution types (MDN, Gaussian, deterministic)
    - CRPS and NLL loss functions for probabilistic training
    - Gradient checkpointing for memory efficiency
    
    Reference:
        Enhanced from "SkyGPT: Probabilistic ultra-short-term solar forecasting"
        Advances in Applied Energy, 2024
    """
    
    def __init__(self, config):
        """
        Initialize SkyGPT-ViTODE model.
        
        Args:
            config: ExperimentConfig object with all hyperparameters
        """
        super().__init__()
        
        self.config = config
        
        # Extract commonly used config values
        self.n_cond_frames = config.data.n_cond_frames
        self.forecast_horizon = config.data.forecast_horizon
        
        # VQ-VAE for discrete latent space
        self.vqvae = VQVAE(
            embedding_dim=config.vqvae.embedding_dim,
            n_codes=config.vqvae.n_codes,
            n_hiddens=config.vqvae.n_hiddens,
            n_res_layers=config.vqvae.n_res_layers,
            downsample=tuple(config.vqvae.downsample),
            in_channels=config.data.channels,
            commitment_cost=config.vqvae.commitment_cost,
            decay=config.vqvae.decay,
            epsilon=config.vqvae.epsilon,
            codebook_init=config.vqvae.codebook_init,
            restart_threshold=config.vqvae.restart_threshold,
            sequence_length=config.data.sequence_length,
            resolution=config.data.resolution
        )
        
        # Neural ODE for continuous dynamics (optional)
        self.neural_ode = None
        if TORCHDIFFEQ_AVAILABLE:
            self.neural_ode = NeuralODEDynamics(
                hidden_dim=config.transformer.hidden_dim,
                n_layers=config.neural_ode.n_layers,
                solver=config.neural_ode.solver,
                rtol=config.neural_ode.rtol,
                atol=config.neural_ode.atol,
                adjoint=config.neural_ode.adjoint,
                activation=config.neural_ode.activation,
                max_num_steps=config.neural_ode.max_num_steps
            )
        
        # Physics-constrained PhyCell
        self.phycell = PhyCell(
            input_dim=config.transformer.hidden_dim,
            hidden_dim=config.neural_ode.physics_kernel_size ** 2,
            n_layers=1,
            kernel_size=config.neural_ode.physics_kernel_size,
            use_physics_constraint=config.neural_ode.use_physics_constraint
        )
        
        # Latent space projections
        self.fc_in = nn.Linear(
            config.vqvae.embedding_dim,
            config.transformer.hidden_dim,
            bias=False
        )
        self.fc_out = nn.Linear(
            config.transformer.hidden_dim,
            config.vqvae.n_codes,
            bias=False
        )
        
        # Vision Transformer for PV prediction
        self.vit_predictor = VisionTransformerPVPredictor(
            img_size=config.data.resolution,
            patch_size=config.vit.patch_size,
            in_channels=config.data.channels,
            embed_dim=config.vit.embed_dim,
            depth=config.vit.depth,
            n_heads=config.vit.n_heads,
            mlp_ratio=config.vit.mlp_ratio,
            qkv_bias=config.vit.qkv_bias,
            drop_rate=config.vit.proj_drop,
            attn_drop_rate=config.vit.attn_drop,
            drop_path_rate=config.vit.drop_path,
            use_temporal_attention=config.vit.use_temporal_attention,
            temporal_depth=config.vit.temporal_depth,
            output_type=config.vit.output_type,
            n_mixture_components=config.vit.n_mixture_components,
            init_scale=config.vit.init_scale
        )
        
        # Conformal predictor for calibration
        self.conformal = ConformalPredictor(
            coverage_level=config.conformal.coverage_level,
            adaptive=config.conformal.use_adaptive,
            n_neighbors=config.conformal.n_neighbors
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.transformer.hidden_dim)
        
        # Integration time points
        t_span = config.neural_ode.t_span
        self.register_buffer(
            't_span',
            torch.linspace(t_span[0], t_span[1], config.neural_ode.integration_times)
        )
        
        # Initialize projections
        nn.init.normal_(self.fc_in.weight, std=0.02)
        nn.init.zeros_(self.fc_out.weight)
    
    def load_vqvae(self, checkpoint_path: str, strict: bool = True):
        """
        Load pretrained VQ-VAE weights.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to require exact key matching
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Extract VQ-VAE weights
        vqvae_state = {}
        for k, v in state_dict.items():
            if k.startswith('vqvae.'):
                vqvae_state[k.replace('vqvae.', '')] = v
            elif not any(k.startswith(p) for p in ['neural_ode.', 'phycell.', 'vit_predictor.']):
                vqvae_state[k] = v
        
        if not vqvae_state:
            vqvae_state = state_dict
        
        self.vqvae.load_state_dict(vqvae_state, strict=strict)
        
        # Optionally freeze VQ-VAE
        if self.config.vqvae.freeze_weights:
            for param in self.vqvae.parameters():
                param.requires_grad = False
            self.vqvae.codebook._need_init = False
            self.vqvae.eval()
    
    def encode_frames(
        self,
        frames: Tensor,
        no_grad: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode video frames to latent representations.
        
        Args:
            frames: Video frames (B, C, T, H, W)
            no_grad: Whether to disable gradients
            
        Returns:
            Tuple of (encoding_indices, continuous_embeddings)
        """
        if no_grad:
            with torch.no_grad():
                return self.vqvae.encode(frames, include_embeddings=True)
        return self.vqvae.encode(frames, include_embeddings=True)
    
    def predict_future_latents(
        self,
        cond_embeddings: Tensor,
        use_checkpointing: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predict future latent states.
        
        Args:
            cond_embeddings: Conditioning embeddings (B, C, T, H, W)
            use_checkpointing: Whether to use gradient checkpointing
            
        Returns:
            Tuple of (logits, physics_loss, kinetic_loss)
        """
        # Rearrange: (B, C, T, H, W) -> (B, T, H, W, C)
        embeddings = shift_dim(cond_embeddings, 1, -1)
        B, T, H, W, C = embeddings.shape
        
        # Project to transformer hidden dim
        h = self.fc_in(embeddings)  # (B, T, H, W, hidden_dim)
        
        # Process through PhyCell
        h_spatial = h.permute(0, 1, 4, 2, 3)  # (B, T, hidden_dim, H, W)
        h_spatial = h_spatial.reshape(B * T, -1, H, W)
        
        phycell_out, physics_loss = self._run_phycell(h_spatial, B, T)
        
        # Reshape back
        hidden_dim = self.config.transformer.hidden_dim
        phycell_out = phycell_out.reshape(B, T, hidden_dim, H, W)
        phycell_out = phycell_out.permute(0, 1, 3, 4, 2)  # (B, T, H, W, hidden_dim)
        
        # Neural ODE for continuous dynamics
        kinetic_loss = torch.tensor(0.0, device=h.device)
        
        if self.neural_ode is not None:
            # Use last timestep as initial condition
            h0 = phycell_out[:, -1].reshape(B, -1)
            
            # Integrate forward
            h_trajectory, h_final = self.neural_ode(h0, self.t_span, return_trajectory=True)
            
            # Compute kinetic energy regularization
            kinetic_loss = self.neural_ode.compute_kinetic_energy(h_trajectory, self.t_span)
            
            # Use final state
            ode_out = h_final.reshape(B, H, W, hidden_dim)
        else:
            ode_out = phycell_out[:, -1]
        
        # Normalize and project to codebook logits
        combined = self.norm(ode_out)
        logits = self.fc_out(combined)  # (B, H, W, n_codes)
        
        return logits, physics_loss, kinetic_loss
    
    def _run_phycell(
        self,
        h_spatial: Tensor,
        batch_size: int,
        n_timesteps: int
    ) -> Tuple[Tensor, Tensor]:
        """Run PhyCell with proper hidden state management."""
        H, W = h_spatial.shape[-2:]
        hidden_dim = h_spatial.shape[1]
        
        # Reshape for sequential processing
        h_seq = h_spatial.reshape(batch_size, n_timesteps, hidden_dim, H, W)
        
        outputs = []
        hidden_states = None
        
        for t in range(n_timesteps):
            out, hidden_states = self.phycell(h_seq[:, t], hidden_states)
            outputs.append(out)
        
        output_seq = torch.stack(outputs, dim=1)  # (B, T, hidden_dim, H, W)
        physics_loss = self.phycell.compute_physics_loss()
        
        return output_seq.reshape(batch_size * n_timesteps, hidden_dim, H, W), physics_loss
    
    def predict_pv(
        self,
        sky_images: Tensor,
        temporal_context: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Predict PV power from sky images.
        
        Args:
            sky_images: Sky images (B, C, H, W) or (B, C, T, H, W)
            temporal_context: Optional historical embeddings
            
        Returns:
            Prediction dictionary
        """
        # Use last frame if video input
        if sky_images.dim() == 5:
            sky_images = sky_images[:, :, -1]
        
        return self.vit_predictor(sky_images, temporal_context)
    
    def forward(
        self,
        video: Tensor,
        pv_target: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Full forward pass for training.
        
        Args:
            video: Input video (B, C, T, H, W)
            pv_target: Target PV output (B,)
            
        Returns:
            Dictionary with losses and predictions
        """
        B, C, T, H, W = video.shape
        device = video.device
        
        # Split conditioning and future frames
        cond_frames = video[:, :, :self.n_cond_frames]
        future_frames = video[:, :, self.n_cond_frames:]
        
        # Encode all frames
        with torch.no_grad() if self.config.vqvae.freeze_weights else torch.enable_grad():
            encodings, embeddings = self.encode_frames(video, no_grad=self.config.vqvae.freeze_weights)
            cond_embeddings = embeddings[:, :, :self.n_cond_frames]
        
        # Predict future latents
        logits, physics_loss, kinetic_loss = self.predict_future_latents(cond_embeddings)
        
        # Video prediction loss (cross-entropy over codebook)
        future_encodings = encodings[:, self.n_cond_frames:]
        
        # Reshape for cross-entropy: (B*H'*W', n_codes) vs (B*H'*W',)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = future_encodings[:, -1].reshape(-1)  # Use last predicted frame
        
        video_loss = F.cross_entropy(logits_flat, targets_flat)
        
        # PV prediction
        pv_output = self.predict_pv(future_frames[:, :, -1])
        
        # PV losses
        if pv_target is not None:
            pv_losses = self.vit_predictor.compute_loss(
                pv_output, pv_target,
                crps_weight=self.config.loss.pv_crps_weight
            )
            pv_loss = pv_losses['total']
        else:
            pv_loss = torch.tensor(0.0, device=device)
            pv_losses = {'total': pv_loss}
        
        # Combined loss
        cfg = self.config.loss
        total_loss = (
            cfg.cross_entropy_weight * video_loss +
            cfg.moment_constraint_weight * physics_loss +
            cfg.kinetic_energy_weight * kinetic_loss +
            cfg.pv_mse_weight * pv_loss
        )
        
        return {
            'loss': total_loss,
            'video_loss': video_loss,
            'physics_loss': physics_loss,
            'kinetic_loss': kinetic_loss,
            'pv_loss': pv_loss,
            'pv_losses': pv_losses,
            'pv_output': pv_output,
            'logits': logits
        }
    
    def generate_samples(
        self,
        conditioning_frames: Tensor,
        n_samples: int = 1,
        temperature: float = 1.0
    ) -> Tensor:
        """
        Generate future frame samples.
        
        Args:
            conditioning_frames: Historical frames (B, C, T, H, W)
            n_samples: Number of samples to generate
            temperature: Sampling temperature
            
        Returns:
            Generated samples (n_samples, B, C, T', H, W)
        """
        B = conditioning_frames.shape[0]
        
        # Encode conditioning
        _, cond_embeddings = self.encode_frames(conditioning_frames)
        
        # Predict future latents
        logits, _, _ = self.predict_future_latents(cond_embeddings)
        
        # Sample from categorical
        probs = F.softmax(logits / temperature, dim=-1)
        
        samples = []
        for _ in range(n_samples):
            # Sample indices
            flat_probs = probs.reshape(-1, probs.shape[-1])
            sampled_idx = torch.multinomial(flat_probs, 1).squeeze(-1)
            sampled_idx = sampled_idx.reshape(B, *logits.shape[1:-1])
            
            # Decode
            decoded = self.vqvae.decode(sampled_idx)
            samples.append(decoded)
        
        return torch.stack(samples, dim=0)
    
    def get_prediction_intervals(
        self,
        sky_images: Tensor,
        coverage_levels: Optional[List[float]] = None
    ) -> Dict[str, Tensor]:
        """
        Get calibrated prediction intervals.
        
        Args:
            sky_images: Input images (B, C, H, W)
            coverage_levels: List of coverage levels (uses default if None)
            
        Returns:
            Dictionary with intervals at each coverage level
        """
        # Get predictions
        pv_output = self.predict_pv(sky_images)
        predictions = pv_output['mean']
        uncertainties = pv_output['std']
        
        # Get intervals from conformal predictor
        lower, upper = self.conformal.get_intervals(predictions, uncertainties)
        
        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'lower': lower,
            'upper': upper,
            'pv_output': pv_output
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_model(config) -> SkyGPTViTODE:
    """
    Factory function to create SkyGPT-ViTODE model.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Initialized model
    """
    model = SkyGPTViTODE(config)
    
    # Load pretrained VQ-VAE if specified
    if config.vqvae.pretrained_path is not None:
        model.load_vqvae(config.vqvae.pretrained_path)
    
    return model


def create_vqvae(config) -> VQVAE:
    """
    Factory function to create standalone VQ-VAE.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Initialized VQ-VAE
    """
    return VQVAE(
        embedding_dim=config.vqvae.embedding_dim,
        n_codes=config.vqvae.n_codes,
        n_hiddens=config.vqvae.n_hiddens,
        n_res_layers=config.vqvae.n_res_layers,
        downsample=tuple(config.vqvae.downsample),
        in_channels=config.data.channels,
        commitment_cost=config.vqvae.commitment_cost,
        decay=config.vqvae.decay,
        epsilon=config.vqvae.epsilon,
        codebook_init=config.vqvae.codebook_init,
        restart_threshold=config.vqvae.restart_threshold,
        sequence_length=config.data.sequence_length,
        resolution=config.data.resolution
    )


def create_vit_predictor(config) -> VisionTransformerPVPredictor:
    """
    Factory function to create standalone ViT predictor.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Initialized ViT predictor
    """
    return VisionTransformerPVPredictor(
        img_size=config.data.resolution,
        patch_size=config.vit.patch_size,
        in_channels=config.data.channels,
        embed_dim=config.vit.embed_dim,
        depth=config.vit.depth,
        n_heads=config.vit.n_heads,
        mlp_ratio=config.vit.mlp_ratio,
        qkv_bias=config.vit.qkv_bias,
        drop_rate=config.vit.proj_drop,
        attn_drop_rate=config.vit.attn_drop,
        drop_path_rate=config.vit.drop_path,
        use_temporal_attention=config.vit.use_temporal_attention,
        temporal_depth=config.vit.temporal_depth,
        output_type=config.vit.output_type,
        n_mixture_components=config.vit.n_mixture_components,
        init_scale=config.vit.init_scale
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("=" * 70)
    print("SkyGPT-ViTODE Model Components Testing")
    print("=" * 70)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Try to import config, create minimal if not available
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from config import ExperimentConfig
        config = ExperimentConfig()
    except ImportError:
        print("Config not found, using minimal test configuration")
        
        from dataclasses import dataclass
        from typing import Tuple
        
        @dataclass
        class DataConfig:
            sequence_length: int = 16
            n_cond_frames: int = 8
            forecast_horizon: int = 8
            resolution: int = 64
            channels: int = 3
        
        @dataclass
        class VQVAEConfig:
            embedding_dim: int = 256
            n_codes: int = 2048
            n_hiddens: int = 240
            n_res_layers: int = 4
            downsample: Tuple[int, int, int] = (4, 4, 4)
            commitment_cost: float = 0.25
            decay: float = 0.99
            epsilon: float = 1e-5
            codebook_init: str = "uniform"
            restart_threshold: float = 1.0
            pretrained_path: str = None
            freeze_weights: bool = False
        
        @dataclass
        class NeuralODEConfig:
            solver: str = "dopri5"
            rtol: float = 1e-3
            atol: float = 1e-4
            adjoint: bool = True
            max_num_steps: int = 1000
            hidden_dim: int = 512
            n_layers: int = 3
            activation: str = "softplus"
            integration_times: int = 8
            t_span: Tuple[float, float] = (0.0, 1.0)
            use_physics_constraint: bool = True
            physics_kernel_size: int = 7
        
        @dataclass
        class ViTConfig:
            patch_size: int = 8
            embed_dim: int = 384
            depth: int = 6
            n_heads: int = 6
            mlp_ratio: float = 4.0
            qkv_bias: bool = True
            attn_drop: float = 0.0
            proj_drop: float = 0.0
            drop_path: float = 0.1
            use_temporal_attention: bool = True
            temporal_depth: int = 2
            output_type: str = "mdn"
            n_mixture_components: int = 5
            init_scale: float = 0.02
        
        @dataclass
        class TransformerConfig:
            hidden_dim: int = 576
        
        @dataclass
        class ConformalConfig:
            coverage_level: float = 0.90
            use_adaptive: bool = False
            n_neighbors: int = 100
        
        @dataclass
        class LossConfig:
            cross_entropy_weight: float = 1.0
            moment_constraint_weight: float = 0.01
            kinetic_energy_weight: float = 0.01
            pv_mse_weight: float = 1.0
            pv_crps_weight: float = 0.5
        
        @dataclass
        class MinimalConfig:
            data: DataConfig = None
            vqvae: VQVAEConfig = None
            neural_ode: NeuralODEConfig = None
            vit: ViTConfig = None
            transformer: TransformerConfig = None
            conformal: ConformalConfig = None
            loss: LossConfig = None
            
            def __post_init__(self):
                self.data = DataConfig()
                self.vqvae = VQVAEConfig()
                self.neural_ode = NeuralODEConfig()
                self.vit = ViTConfig()
                self.transformer = TransformerConfig()
                self.conformal = ConformalConfig()
                self.loss = LossConfig()
        
        config = MinimalConfig()
    
    # Test individual components
    print("\n1. Testing VQ-VAE...")
    vqvae = VQVAE(
        embedding_dim=256,
        n_codes=2048,
        n_hiddens=128,  # Reduced for testing
        n_res_layers=2,
        downsample=(4, 4, 4),
        in_channels=3,
        sequence_length=16,
        resolution=64
    ).to(device)
    
    dummy_video = torch.randn(2, 3, 16, 64, 64).to(device)
    recon_loss, recon, vq_out = vqvae(dummy_video)
    
    print(f"   Input: {dummy_video.shape}")
    print(f"   Reconstruction: {recon.shape}")
    print(f"   Latent shape: {vqvae.latent_shape}")
    print(f"   Recon loss: {recon_loss.item():.4f}")
    print(f"   Perplexity: {vq_out['perplexity'].item():.2f}")
    print("   ✓ VQ-VAE test passed")
    
    # Test MDN
    print("\n2. Testing Mixture Density Network...")
    mdn = MixtureDensityNetwork(input_dim=384, n_components=5).to(device)
    dummy_features = torch.randn(4, 384).to(device)
    mdn_output = mdn(dummy_features)
    
    print(f"   Weights: {mdn_output['weights'].shape}")
    print(f"   Means: {mdn_output['means'].shape}")
    print(f"   Scales: {mdn_output['scales'].shape}")
    
    samples = mdn.sample(mdn_output, n_samples=10)
    print(f"   Samples: {samples.shape}")
    
    target = torch.randn(4).to(device)
    nll = mdn.nll_loss(mdn_output, target)
    crps = mdn.crps_loss(mdn_output, target)
    print(f"   NLL loss: {nll.item():.4f}")
    print(f"   CRPS loss: {crps.item():.4f}")
    print("   ✓ MDN test passed")
    
    # Test ViT
    print("\n3. Testing Vision Transformer...")
    vit = VisionTransformerPVPredictor(
        img_size=64,
        patch_size=8,
        embed_dim=256,
        depth=4,
        n_heads=4,
        output_type="mdn",
        n_mixture_components=5
    ).to(device)
    
    dummy_image = torch.randn(2, 3, 64, 64).to(device)
    vit_output = vit(dummy_image)
    
    print(f"   Input: {dummy_image.shape}")
    print(f"   Output keys: {list(vit_output.keys())}")
    print(f"   Mean prediction: {vit_output['mean'].shape}")
    print(f"   Std prediction: {vit_output['std'].shape}")
    print("   ✓ ViT test passed")
    
    # Test PhyCell
    print("\n4. Testing PhyCell...")
    phycell = PhyCell(
        input_dim=64,
        hidden_dim=49,
        kernel_size=7,
        use_physics_constraint=True
    ).to(device)
    
    dummy_seq = torch.randn(2, 8, 64, 16, 16).to(device)
    phy_out, phy_loss = phycell.forward_sequence(dummy_seq)
    
    print(f"   Input: {dummy_seq.shape}")
    print(f"   Output: {phy_out.shape}")
    print(f"   Physics loss: {phy_loss.item():.4f}")
    print("   ✓ PhyCell test passed")
    
    # Test Neural ODE (if available)
    if TORCHDIFFEQ_AVAILABLE:
        print("\n5. Testing Neural ODE...")
        neural_ode = NeuralODEDynamics(
            hidden_dim=64,
            n_layers=2,
            solver="euler"
        ).to(device)
        
        h0 = torch.randn(2, 64).to(device)
        t_span = torch.linspace(0, 1, 8).to(device)
        
        h_traj, h_final = neural_ode(h0, t_span, return_trajectory=True)
        ke = neural_ode.compute_kinetic_energy(h_traj, t_span)
        
        print(f"   Initial: {h0.shape}")
        print(f"   Trajectory: {h_traj.shape}")
        print(f"   Final: {h_final.shape}")
        print(f"   NFE: {neural_ode.nfe}")
        print(f"   Kinetic energy: {ke.item():.4f}")
        print("   ✓ Neural ODE test passed")
    else:
        print("\n5. Skipping Neural ODE (torchdiffeq not installed)")
    
    # Count parameters
    print("\n6. Parameter counts:")
    vqvae_params = count_parameters(vqvae)
    vit_params = count_parameters(vit)
    
    print(f"   VQ-VAE: {vqvae_params['total_millions']:.2f}M total, "
          f"{vqvae_params['trainable_millions']:.2f}M trainable")
    print(f"   ViT: {vit_params['total_millions']:.2f}M total, "
          f"{vit_params['trainable_millions']:.2f}M trainable")
    
    # Clean up
    del vqvae, mdn, vit, phycell
    if TORCHDIFFEQ_AVAILABLE:
        del neural_ode
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\n" + "=" * 70)
    print("✓ All component tests passed successfully!")
    print("=" * 70)