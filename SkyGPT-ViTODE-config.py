"""
SkyGPT-ViTODE Configuration Management System
==============================================

This module provides a comprehensive configuration system for the SkyGPT-ViTODE
framework, enabling reproducible experiments and systematic hyperparameter management.

The configuration system supports:
    - Hierarchical dataclass-based configuration with full type safety
    - YAML serialization and deserialization with proper tuple handling
    - Configuration hashing for experiment tracking and reproducibility
    - Comprehensive validation of all hyperparameters
    - Predefined ablation study configurations
    - Command-line override support
"""

from __future__ import annotations

import os
import sys
import yaml
import json
import hashlib
import logging
import random
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass, field, asdict, fields
from typing import (
    Optional, List, Tuple, Dict, Any, Union, 
    TypeVar, Type, get_type_hints, get_origin, get_args
)

import numpy as np

# Conditional PyTorch import for configuration-only usage
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. GPU validation and seed setting will be disabled.",
        ImportWarning
    )


# ============================================================================
# Constants
# ============================================================================

# Valid options for enumerated configuration fields
VALID_ODE_SOLVERS = frozenset({"dopri5", "euler", "rk4", "adaptive_heun", "midpoint", "bosh3"})
VALID_OPTIMIZERS = frozenset({"adam", "adamw", "sgd", "rmsprop"})
VALID_SCHEDULERS = frozenset({"cosine", "step", "plateau", "warmup_cosine", "linear", "one_cycle"})
VALID_ACTIVATIONS = frozenset({"softplus", "tanh", "relu", "gelu", "silu", "elu"})
VALID_OUTPUT_TYPES = frozenset({"deterministic", "gaussian", "mdn"})
VALID_ATTENTION_TYPES = frozenset({"full", "sparse", "linear"})
VALID_AMP_DTYPES = frozenset({"float16", "bfloat16"})
VALID_FIGURE_FORMATS = frozenset({"pdf", "png", "svg", "eps"})

# Version for configuration compatibility checking
CONFIG_VERSION = "1.1.0"


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(
    log_dir: str, 
    experiment_name: str,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO
) -> logging.Logger:
    """
    Configure logging for experiment tracking.
    
    Creates both file and console handlers with appropriate formatting.
    Prevents duplicate handlers when called multiple times.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name of the current experiment
        level: Logging level for file handler
        console_level: Logging level for console handler
        
    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{experiment_name}_{timestamp}.log"
    
    # Configure logging format
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get or create logger
    logger = logging.getLogger('SkyGPT-ViTODE')
    
    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# ============================================================================
# YAML Custom Handlers for Tuple Serialization
# ============================================================================

def _tuple_representer(dumper: yaml.Dumper, data: tuple) -> yaml.Node:
    """Custom YAML representer for tuples to preserve type information."""
    return dumper.represent_sequence('!tuple', list(data))


def _tuple_constructor(loader: yaml.Loader, node: yaml.Node) -> tuple:
    """Custom YAML constructor for tuples."""
    return tuple(loader.construct_sequence(node))


# Register custom handlers
yaml.add_representer(tuple, _tuple_representer)
yaml.add_constructor('!tuple', _tuple_constructor, Loader=yaml.SafeLoader)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Defines all parameters related to the SKIPP'D dataset handling,
    including paths, sequence structure, augmentation, and normalization.
    """
    
    # Dataset paths
    data_path: str = "./data/skippd/2017_2019_images_pv_processed.hdf5"
    cache_dir: str = "./data/cache"
    
    # Sequence configuration
    sequence_length: int = 16          # Total frames (n_cond_frames + forecast_horizon)
    n_cond_frames: int = 8             # Number of conditioning (historical) frames
    forecast_horizon: int = 8          # Number of frames to predict
    temporal_stride: int = 2           # Minutes between consecutive frames
    
    # Image configuration
    resolution: int = 64               # Spatial resolution (height = width)
    channels: int = 3                  # Number of image channels (RGB)
    
    # Data splits (must sum to 1.0)
    train_ratio: float = 0.88          # Training set ratio
    val_ratio: float = 0.07            # Validation set ratio
    test_ratio: float = 0.05           # Test set ratio (held-out cloudy days)
    
    # Test set specification (November-December 2019 cloudy days as in original paper)
    test_start_date: str = "2019-11-01"
    test_end_date: str = "2019-12-31"
    
    # Cloud filtering
    cloudy_only: bool = True           # Use only cloudy samples for training/evaluation
    cloud_threshold: float = 0.3       # Normalized red-blue ratio threshold for cloud detection
    
    # DataLoader configuration
    batch_size: int = 32               # Batch size per GPU
    num_workers: int = 8               # Number of data loading workers per GPU
    pin_memory: bool = True            # Pin memory for faster GPU transfer
    prefetch_factor: int = 2           # Number of batches to prefetch per worker
    persistent_workers: bool = True    # Keep workers alive between epochs
    drop_last: bool = True             # Drop incomplete final batch (for stable batch norm)
    
    # Data augmentation (applied during training only)
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5  # Probability of horizontal flip
    rotation_degrees: float = 15.0     # Maximum rotation angle in degrees
    brightness_jitter: float = 0.1     # Brightness variation factor
    contrast_jitter: float = 0.1       # Contrast variation factor
    
    # Normalization statistics
    # Note: Using ImageNet statistics as common practice for transfer learning.
    # For SKIPP'D-specific normalization, compute from training set.
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    pv_max: float = 30.0               # Maximum PV output (kW) for normalization
    pv_min: float = 0.0                # Minimum PV output (kW)
    
    def __post_init__(self):
        """Validate data configuration parameters."""
        # Validate ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got {total:.6f}")
        
        # Validate sequence consistency
        if self.sequence_length != self.n_cond_frames + self.forecast_horizon:
            raise ValueError(
                f"sequence_length ({self.sequence_length}) must equal "
                f"n_cond_frames ({self.n_cond_frames}) + forecast_horizon ({self.forecast_horizon})"
            )
        
        # Validate image dimensions
        if self.resolution <= 0 or self.resolution % 8 != 0:
            raise ValueError(f"resolution must be positive and divisible by 8, got {self.resolution}")
        
        if self.channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {self.channels}")
        
        # Validate augmentation parameters
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError(f"horizontal_flip_prob must be in [0, 1], got {self.horizontal_flip_prob}")
        
        if self.rotation_degrees < 0:
            raise ValueError(f"rotation_degrees must be non-negative, got {self.rotation_degrees}")


@dataclass
class VQVAEConfig:
    """
    Configuration for Vector Quantized Variational Autoencoder.
    
    The VQ-VAE learns discrete latent representations of sky video sequences,
    enabling efficient compression and generation.
    """
    
    # Architecture parameters
    embedding_dim: int = 256           # Dimension of codebook embeddings
    n_codes: int = 2048                # Number of codebook entries (vocabulary size)
    n_hiddens: int = 240               # Hidden dimension in encoder/decoder convolutions
    n_res_layers: int = 4              # Number of residual blocks in encoder/decoder
    downsample: Tuple[int, int, int] = (4, 4, 4)  # Downsampling factors (T, H, W)
    
    # Training parameters
    commitment_cost: float = 0.25      # Weight for commitment loss (β in VQ-VAE paper)
    decay: float = 0.99                # EMA decay rate for codebook update
    epsilon: float = 1e-5              # Epsilon for numerical stability in EMA
    
    # Codebook initialization
    codebook_init: str = "uniform"     # Initialization: "uniform", "normal", "kmeans"
    restart_threshold: float = 1.0     # Reset unused codes with usage below this threshold
    
    # Pretrained weights
    pretrained_path: Optional[str] = None  # Path to pretrained VQ-VAE checkpoint
    freeze_weights: bool = True        # Freeze VQ-VAE weights during full model training
    
    def __post_init__(self):
        """Validate VQ-VAE configuration."""
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        
        if self.n_codes <= 0:
            raise ValueError(f"n_codes must be positive, got {self.n_codes}")
        
        if not all(d > 0 and (d & (d - 1) == 0) for d in self.downsample):
            raise ValueError(f"downsample factors must be positive powers of 2, got {self.downsample}")
        
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {self.decay}")
        
        if self.commitment_cost < 0:
            raise ValueError(f"commitment_cost must be non-negative, got {self.commitment_cost}")


@dataclass
class NeuralODEConfig:
    """
    Configuration for Neural ODE dynamics module.
    
    The Neural ODE models continuous-time latent dynamics for cloud motion,
    enabling smooth interpolation and physically meaningful extrapolation.
    """
    
    # ODE solver configuration
    solver: str = "dopri5"             # ODE solver algorithm
    rtol: float = 1e-3                 # Relative tolerance for adaptive solvers
    atol: float = 1e-4                 # Absolute tolerance for adaptive solvers
    adjoint: bool = True               # Use adjoint method for memory-efficient backprop
    
    # Network architecture
    hidden_dim: int = 512              # Hidden dimension in ODE function network
    n_layers: int = 3                  # Number of layers in ODE function
    activation: str = "softplus"       # Activation function
    
    # Time integration
    integration_times: int = 8         # Number of time points for ODE evaluation
    t_span: Tuple[float, float] = (0.0, 1.0)  # Normalized time span for integration
    
    # Physics constraints (moment constraints for PDE approximation)
    use_physics_constraint: bool = True
    physics_kernel_size: int = 7       # Kernel size for moment constraint computation
    physics_weight: float = 0.01       # Weight for physics constraint loss
    
    # Regularization
    kinetic_energy_reg: float = 0.01   # Kinetic energy regularization weight
    jacobian_reg: float = 0.0          # Jacobian Frobenius norm regularization
    
    # Numerical stability
    max_num_steps: int = 1000          # Maximum integration steps (prevents infinite loops)
    
    def __post_init__(self):
        """Validate Neural ODE configuration."""
        if self.solver not in VALID_ODE_SOLVERS:
            raise ValueError(f"solver must be one of {VALID_ODE_SOLVERS}, got '{self.solver}'")
        
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(f"activation must be one of {VALID_ACTIVATIONS}, got '{self.activation}'")
        
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError(f"rtol and atol must be positive, got rtol={self.rtol}, atol={self.atol}")
        
        if self.hidden_dim <= 0 or self.n_layers <= 0:
            raise ValueError(f"hidden_dim and n_layers must be positive")
        
        if len(self.t_span) != 2 or self.t_span[0] >= self.t_span[1]:
            raise ValueError(f"t_span must be (start, end) with start < end, got {self.t_span}")
        
        if self.physics_kernel_size < 3 or self.physics_kernel_size % 2 == 0:
            raise ValueError(f"physics_kernel_size must be odd and >= 3, got {self.physics_kernel_size}")


@dataclass
class ViTConfig:
    """
    Configuration for Vision Transformer PV predictor.
    
    The ViT processes predicted future sky images to generate
    probabilistic PV power output forecasts.
    """
    
    # Patch embedding
    patch_size: int = 8                # Patch size (creates (resolution/patch_size)² patches)
    
    # Transformer architecture
    embed_dim: int = 384               # Embedding dimension
    depth: int = 6                     # Number of transformer blocks
    n_heads: int = 6                   # Number of attention heads
    mlp_ratio: float = 4.0             # MLP hidden dimension ratio (mlp_dim = embed_dim * mlp_ratio)
    
    # Attention configuration
    qkv_bias: bool = True              # Include bias in QKV projection
    attn_drop: float = 0.0             # Attention weights dropout
    proj_drop: float = 0.0             # Output projection dropout
    drop_path: float = 0.1             # Stochastic depth drop rate
    
    # Temporal cross-attention for historical context
    use_temporal_attention: bool = True
    temporal_depth: int = 2            # Number of temporal cross-attention layers
    
    # Output head configuration
    output_type: str = "mdn"           # Output type: "deterministic", "gaussian", "mdn"
    n_mixture_components: int = 5      # Number of MDN mixture components
    
    # Regularization
    label_smoothing: float = 0.0       # Label smoothing for classification tasks
    weight_decay: float = 0.05         # Weight decay for ViT parameters
    
    # Initialization
    init_scale: float = 0.02           # Standard deviation for weight initialization
    
    def __post_init__(self):
        """Validate ViT configuration."""
        if self.output_type not in VALID_OUTPUT_TYPES:
            raise ValueError(f"output_type must be one of {VALID_OUTPUT_TYPES}, got '{self.output_type}'")
        
        if self.embed_dim <= 0 or self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be positive and divisible by n_heads ({self.n_heads})"
            )
        
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        
        if self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        
        if self.output_type == "mdn" and self.n_mixture_components <= 0:
            raise ValueError(f"n_mixture_components must be positive for MDN, got {self.n_mixture_components}")
        
        if not 0.0 <= self.drop_path < 1.0:
            raise ValueError(f"drop_path must be in [0, 1), got {self.drop_path}")


@dataclass
class ConformalConfig:
    """
    Configuration for conformal prediction calibration.
    
    Conformal prediction provides distribution-free coverage guarantees
    for prediction intervals, ensuring reliable uncertainty quantification.
    """
    
    # Primary conformal settings
    coverage_level: float = 0.90       # Target coverage probability (1 - α)
    calibration_ratio: float = 0.15    # Ratio of validation set used for calibration
    
    # Adaptive conformal prediction
    use_adaptive: bool = True          # Use locally adaptive conformal prediction
    n_neighbors: int = 100             # Number of neighbors for local calibration
    distance_metric: str = "euclidean" # Distance metric for neighbor computation
    
    # Multiple coverage levels for comprehensive analysis
    coverage_levels: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)
    
    # Conformity score type
    score_type: str = "absolute"       # Score type: "absolute", "normalized", "quantile"
    
    def __post_init__(self):
        """Validate conformal prediction configuration."""
        if not 0.0 < self.coverage_level < 1.0:
            raise ValueError(f"coverage_level must be in (0, 1), got {self.coverage_level}")
        
        if not 0.0 < self.calibration_ratio < 1.0:
            raise ValueError(f"calibration_ratio must be in (0, 1), got {self.calibration_ratio}")
        
        if not all(0.0 < cl < 1.0 for cl in self.coverage_levels):
            raise ValueError(f"All coverage_levels must be in (0, 1), got {self.coverage_levels}")
        
        if self.n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {self.n_neighbors}")


@dataclass  
class TransformerConfig:
    """
    Configuration for the GPT-style transformer in video prediction.
    
    This transformer operates on discrete latent codes to predict
    future sky image sequences autoregressively.
    """
    
    # Architecture
    hidden_dim: int = 576              # Hidden/embedding dimension
    n_heads: int = 4                   # Number of attention heads
    n_layers: int = 8                  # Number of transformer layers
    
    # Dropout rates
    dropout: float = 0.2               # General dropout rate
    attn_dropout: float = 0.3          # Attention-specific dropout
    embed_dropout: float = 0.1         # Embedding dropout
    
    # Attention configuration
    attn_type: str = "full"            # Attention type: "full", "sparse", "linear"
    
    # Class conditioning (optional, not used for SKIPP'D)
    class_cond: bool = False           # Enable class conditioning
    class_cond_dim: int = 0            # Dimension of class condition embedding
    n_classes: int = 0                 # Number of classes for conditioning
    
    def __post_init__(self):
        """Validate transformer configuration."""
        if self.attn_type not in VALID_ATTENTION_TYPES:
            raise ValueError(f"attn_type must be one of {VALID_ATTENTION_TYPES}, got '{self.attn_type}'")
        
        if self.hidden_dim <= 0 or self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be positive and divisible by n_heads ({self.n_heads})"
            )
        
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        
        for name, value in [("dropout", self.dropout), ("attn_dropout", self.attn_dropout)]:
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")


@dataclass
class TrainingConfig:
    """
    Configuration for training procedure.
    
    Defines optimization, scheduling, distributed training,
    and checkpointing parameters.
    """
    
    # Optimizer configuration
    optimizer: str = "adamw"           # Optimizer algorithm
    learning_rate: float = 3e-4        # Initial/peak learning rate
    weight_decay: float = 0.01         # Weight decay (L2 regularization)
    betas: Tuple[float, float] = (0.9, 0.999)  # Adam beta parameters
    eps: float = 1e-8                  # Adam epsilon for numerical stability
    
    # Learning rate scheduling
    scheduler: str = "warmup_cosine"   # LR scheduler type
    warmup_epochs: int = 5             # Number of warmup epochs
    warmup_start_lr: float = 1e-6      # Starting LR for warmup
    min_lr: float = 1e-6               # Minimum LR (for cosine decay)
    
    # Training duration
    max_epochs: int = 100              # Maximum number of training epochs
    max_steps: Optional[int] = None    # Max steps (overrides epochs if set)
    
    # Batch configuration
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15                 # Epochs without improvement before stopping
    min_delta: float = 1e-4            # Minimum improvement threshold
    monitor_metric: str = "val_loss"   # Metric to monitor for early stopping
    monitor_mode: str = "min"          # "min" or "max" for monitored metric
    
    # Gradient clipping
    gradient_clip_val: float = 1.0     # Maximum gradient norm
    gradient_clip_algorithm: str = "norm"  # Clipping algorithm: "norm" or "value"
    
    # Mixed precision training
    use_amp: bool = True               # Enable automatic mixed precision
    amp_dtype: str = "float16"         # AMP dtype: "float16" or "bfloat16"
    
    # Distributed training
    distributed: bool = True           # Enable distributed data parallel
    n_gpus: int = 4                    # Number of GPUs to use
    sync_batchnorm: bool = True        # Synchronize batch norm across GPUs
    find_unused_parameters: bool = True  # For flexible model architectures
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_top_k: int = 3                # Number of best checkpoints to keep
    save_every_n_epochs: int = 5       # Save checkpoint every N epochs
    save_last: bool = True             # Always save last checkpoint
    resume_from_checkpoint: Optional[str] = None  # Path to resume training
    
    # Reproducibility
    seed: int = 42                     # Random seed for reproducibility
    deterministic: bool = True         # Use deterministic algorithms
    benchmark: bool = False            # cuDNN benchmark mode (False for reproducibility)
    
    # Logging
    log_every_n_steps: int = 50        # Log metrics every N steps
    val_check_interval: float = 1.0    # Validation frequency (1.0 = every epoch)
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {VALID_OPTIMIZERS}, got '{self.optimizer}'")
        
        if self.scheduler not in VALID_SCHEDULERS:
            raise ValueError(f"scheduler must be one of {VALID_SCHEDULERS}, got '{self.scheduler}'")
        
        if self.amp_dtype not in VALID_AMP_DTYPES:
            raise ValueError(f"amp_dtype must be one of {VALID_AMP_DTYPES}, got '{self.amp_dtype}'")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        if not (0.0 <= self.betas[0] < 1.0 and 0.0 <= self.betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {self.betas}")
        
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")
        
        if self.n_gpus < 0:
            raise ValueError(f"n_gpus must be non-negative, got {self.n_gpus}")
    
    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size accounting for GPUs and accumulation."""
        # Note: Requires DataConfig.batch_size to compute fully
        return self.gradient_accumulation_steps * max(1, self.n_gpus)


@dataclass
class LossConfig:
    """
    Configuration for loss functions and their weights.
    
    Defines the multi-objective loss function combining video prediction,
    physics constraints, and PV forecasting losses.
    """
    
    # Video prediction losses
    reconstruction_weight: float = 1.0     # VQ-VAE reconstruction loss weight
    codebook_weight: float = 1.0           # Codebook/commitment loss weight
    cross_entropy_weight: float = 1.0      # Transformer cross-entropy loss weight
    
    # Physics constraint loss
    moment_constraint_weight: float = 0.01 # PhyCell moment constraint loss
    
    # PV prediction losses
    pv_mse_weight: float = 1.0             # Mean squared error weight
    pv_crps_weight: float = 0.5            # CRPS loss weight (for probabilistic models)
    pv_nll_weight: float = 0.0             # Negative log-likelihood weight
    
    # Regularization losses
    kinetic_energy_weight: float = 0.01    # Neural ODE kinetic energy regularization
    
    # Joint training configuration
    joint_training: bool = True            # Enable end-to-end training
    pv_feedback_weight: float = 0.1        # Weight for PV-aware video generation
    
    # Loss normalization
    normalize_losses: bool = True          # Normalize losses to similar scales
    
    def __post_init__(self):
        """Validate loss configuration."""
        loss_weights = [
            ("reconstruction_weight", self.reconstruction_weight),
            ("codebook_weight", self.codebook_weight),
            ("cross_entropy_weight", self.cross_entropy_weight),
            ("moment_constraint_weight", self.moment_constraint_weight),
            ("pv_mse_weight", self.pv_mse_weight),
            ("pv_crps_weight", self.pv_crps_weight),
            ("kinetic_energy_weight", self.kinetic_energy_weight),
            ("pv_feedback_weight", self.pv_feedback_weight),
        ]
        
        for name, weight in loss_weights:
            if weight < 0:
                raise ValueError(f"{name} must be non-negative, got {weight}")


@dataclass
class EvaluationConfig:
    """
    Configuration for model evaluation.
    
    Defines metrics, visualization settings, and statistical
    testing parameters for comprehensive model assessment.
    """
    
    # Metrics to compute
    compute_crps: bool = True          # Continuous Ranked Probability Score
    compute_winkler: bool = True       # Winkler Score for prediction intervals
    compute_reliability: bool = True   # Reliability diagram data
    compute_sharpness: bool = True     # Prediction interval width analysis
    compute_skill_score: bool = True   # Skill score relative to persistence
    
    # Probabilistic sampling
    n_future_samples: int = 50         # Number of samples for probabilistic forecasts
    
    # Output settings
    save_predictions: bool = True      # Save raw predictions to disk
    save_figures: bool = True          # Generate and save figures
    figure_dpi: int = 300              # DPI for saved figures
    figure_format: str = "pdf"         # Figure format
    
    # Statistical testing
    bootstrap_samples: int = 1000      # Number of bootstrap resamples
    significance_level: float = 0.05   # Significance level for hypothesis tests
    use_bonferroni: bool = True        # Apply Bonferroni correction
    
    # Baseline comparisons
    compare_persistence: bool = True   # Compare with smart persistence model
    compare_climatology: bool = True   # Compare with climatological mean
    
    def __post_init__(self):
        """Validate evaluation configuration."""
        if self.figure_format not in VALID_FIGURE_FORMATS:
            raise ValueError(f"figure_format must be one of {VALID_FIGURE_FORMATS}, got '{self.figure_format}'")
        
        if self.n_future_samples <= 0:
            raise ValueError(f"n_future_samples must be positive, got {self.n_future_samples}")
        
        if self.bootstrap_samples <= 0:
            raise ValueError(f"bootstrap_samples must be positive, got {self.bootstrap_samples}")
        
        if not 0.0 < self.significance_level < 1.0:
            raise ValueError(f"significance_level must be in (0, 1), got {self.significance_level}")


@dataclass
class ExperimentConfig:
    """
    Master configuration combining all sub-configurations.
    
    This class provides the complete experiment specification,
    including configuration validation, serialization, and
    experiment tracking utilities.
    """
    
    # Experiment metadata
    experiment_name: str = "skygpt_vitode"
    description: str = "SkyGPT-ViTODE: Enhanced probabilistic solar forecasting with Vision Transformer and Neural ODE"
    version: str = CONFIG_VERSION
    tags: List[str] = field(default_factory=list)
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    neural_ode: NeuralODEConfig = field(default_factory=NeuralODEConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Global paths
    project_root: str = "."
    log_dir: str = "./logs"
    results_dir: str = "./results"
    
    # Skip validation flag (for loading incomplete configs)
    _skip_validation: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Validate and initialize configuration."""
        if not self._skip_validation:
            self._validate()
        self._create_directories()
    
    def _validate(self):
        """Perform comprehensive configuration validation."""
        # Cross-component validation
        
        # Validate ViT patch size compatibility with resolution
        if self.data.resolution % self.vit.patch_size != 0:
            raise ValueError(
                f"data.resolution ({self.data.resolution}) must be divisible by "
                f"vit.patch_size ({self.vit.patch_size})"
            )
        
        # Validate GPU availability (only if PyTorch available and GPUs requested)
        if TORCH_AVAILABLE and self.training.n_gpus > 0:
            available_gpus = torch.cuda.device_count()
            if self.training.n_gpus > available_gpus:
                warnings.warn(
                    f"Requested {self.training.n_gpus} GPUs but only {available_gpus} available. "
                    f"Training will use {available_gpus} GPUs.",
                    RuntimeWarning
                )
        
        # Validate checkpoint path if resuming
        if self.training.resume_from_checkpoint is not None:
            if not os.path.exists(self.training.resume_from_checkpoint):
                raise ValueError(
                    f"resume_from_checkpoint path does not exist: {self.training.resume_from_checkpoint}"
                )
        
        # Validate VQ-VAE pretrained path if specified
        if self.vqvae.pretrained_path is not None:
            if not os.path.exists(self.vqvae.pretrained_path):
                raise ValueError(
                    f"vqvae.pretrained_path does not exist: {self.vqvae.pretrained_path}"
                )
    
    def _create_directories(self):
        """Create necessary directories for experiment."""
        directories = [
            self.log_dir,
            self.results_dir,
            self.training.checkpoint_dir,
            self.data.cache_dir,
            os.path.join(self.results_dir, "figures"),
            os.path.join(self.results_dir, "tables"),
            os.path.join(self.results_dir, "predictions"),
            os.path.join(self.results_dir, "ablations"),
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                warnings.warn(f"Could not create directory {directory}: {e}")
    
    def get_config_hash(self) -> str:
        """
        Generate unique hash for configuration.
        
        Used for experiment tracking and cache invalidation.
        Excludes path-related fields that don't affect computation.
        
        Returns:
            8-character hexadecimal hash string
        """
        # Create dict excluding volatile fields
        config_dict = asdict(self)
        
        # Remove fields that shouldn't affect the hash
        volatile_fields = [
            'project_root', 'log_dir', 'results_dir', 'checkpoint_dir',
            'cache_dir', 'data_path', 'pretrained_path', 'resume_from_checkpoint',
            '_skip_validation'
        ]
        
        def remove_volatile(d: dict, keys: list) -> dict:
            result = {}
            for k, v in d.items():
                if k in keys:
                    continue
                if isinstance(v, dict):
                    result[k] = remove_volatile(v, keys)
                else:
                    result[k] = v
            return result
        
        filtered_dict = remove_volatile(config_dict, volatile_fields)
        config_str = json.dumps(filtered_dict, sort_keys=True, default=str)
        
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save configuration to YAML file.
        
        Args:
            path: Output file path (auto-generated if None)
            
        Returns:
            Path to saved configuration file
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{self.get_config_hash()}_{timestamp}.yaml"
            path = os.path.join(self.log_dir, filename)
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save
        config_dict = self._to_serializable_dict()
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        return path
    
    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary with tuple preservation."""
        def convert(obj):
            if isinstance(obj, tuple):
                return obj  # Keep as tuple for custom YAML serialization
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items() if not k.startswith('_')}
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items() if not k.startswith('_')}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj
        
        result = {}
        for f in fields(self):
            if f.name.startswith('_'):
                continue
            value = getattr(self, f.name)
            result[f.name] = convert(value)
        
        return result
    
    @classmethod
    def load(cls, path: str, strict: bool = True) -> 'ExperimentConfig':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            strict: If True, validate configuration after loading
            
        Returns:
            Loaded ExperimentConfig instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict, validate=strict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], validate: bool = True) -> 'ExperimentConfig':
        """
        Create configuration from dictionary.
        
        Handles proper reconstruction of nested dataclasses and tuple fields.
        
        Args:
            config_dict: Configuration dictionary
            validate: Whether to validate the configuration
            
        Returns:
            ExperimentConfig instance
        """
        def convert_tuples(obj, type_hint=None):
            """Recursively convert lists to tuples where appropriate."""
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                # Check if this should be a tuple based on content or type hint
                return tuple(convert_tuples(item) for item in obj)
            else:
                return obj
        
        # Convert lists back to tuples in nested dicts
        config_dict = convert_tuples(config_dict)
        
        # Parse sub-configurations with proper tuple handling
        def safe_get(key: str, default_factory):
            raw = config_dict.get(key, {})
            if isinstance(raw, dict):
                return default_factory(**raw)
            return default_factory()
        
        return cls(
            experiment_name=config_dict.get('experiment_name', 'skygpt_vitode'),
            description=config_dict.get('description', ''),
            version=config_dict.get('version', CONFIG_VERSION),
            tags=list(config_dict.get('tags', [])),
            data=safe_get('data', DataConfig),
            vqvae=safe_get('vqvae', VQVAEConfig),
            neural_ode=safe_get('neural_ode', NeuralODEConfig),
            vit=safe_get('vit', ViTConfig),
            conformal=safe_get('conformal', ConformalConfig),
            transformer=safe_get('transformer', TransformerConfig),
            training=safe_get('training', TrainingConfig),
            loss=safe_get('loss', LossConfig),
            evaluation=safe_get('evaluation', EvaluationConfig),
            project_root=config_dict.get('project_root', '.'),
            log_dir=config_dict.get('log_dir', './logs'),
            results_dir=config_dict.get('results_dir', './results'),
            _skip_validation=not validate,
        )
    
    def merge(self, overrides: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create new configuration with overrides applied.
        
        Args:
            overrides: Dictionary of override values (supports nested keys with '.')
            
        Returns:
            New ExperimentConfig with overrides applied
        """
        config_dict = self._to_serializable_dict()
        
        for key, value in overrides.items():
            keys = key.split('.')
            target = config_dict
            
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            target[keys[-1]] = value
        
        return ExperimentConfig.from_dict(config_dict)
    
    def diff(self, other: 'ExperimentConfig') -> Dict[str, Tuple[Any, Any]]:
        """
        Compare this configuration with another.
        
        Args:
            other: Configuration to compare against
            
        Returns:
            Dictionary of differing fields: {field_path: (self_value, other_value)}
        """
        def flatten_dict(d: dict, parent_key: str = '') -> dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        self_flat = flatten_dict(self._to_serializable_dict())
        other_flat = flatten_dict(other._to_serializable_dict())
        
        all_keys = set(self_flat.keys()) | set(other_flat.keys())
        
        diffs = {}
        for key in all_keys:
            self_val = self_flat.get(key)
            other_val = other_flat.get(key)
            if self_val != other_val:
                diffs[key] = (self_val, other_val)
        
        return diffs
    
    def summary(self) -> str:
        """Generate human-readable configuration summary."""
        lines = [
            "=" * 70,
            f"SkyGPT-ViTODE Configuration Summary",
            "=" * 70,
            f"Experiment: {self.experiment_name}",
            f"Version: {self.version}",
            f"Config Hash: {self.get_config_hash()}",
            f"Description: {self.description}",
            "",
            "Data Configuration:",
            f"  Resolution: {self.data.resolution}x{self.data.resolution}",
            f"  Sequence: {self.data.n_cond_frames} conditioning + {self.data.forecast_horizon} forecast frames",
            f"  Batch Size: {self.data.batch_size} per GPU",
            f"  Cloudy Only: {self.data.cloudy_only}",
            "",
            "Model Configuration:",
            f"  VQ-VAE Codes: {self.vqvae.n_codes} x {self.vqvae.embedding_dim}D",
            f"  ViT: {self.vit.depth} layers, {self.vit.embed_dim}D, {self.vit.n_heads} heads",
            f"  Output Type: {self.vit.output_type}" + 
                (f" ({self.vit.n_mixture_components} components)" if self.vit.output_type == "mdn" else ""),
            f"  Neural ODE Solver: {self.neural_ode.solver}",
            f"  Physics Constraints: {self.neural_ode.use_physics_constraint}",
            "",
            "Training Configuration:",
            f"  Max Epochs: {self.training.max_epochs}",
            f"  Learning Rate: {self.training.learning_rate}",
            f"  Optimizer: {self.training.optimizer}",
            f"  Scheduler: {self.training.scheduler}",
            f"  GPUs: {self.training.n_gpus}",
            f"  Mixed Precision: {self.training.use_amp} ({self.training.amp_dtype})",
            f"  Gradient Accumulation: {self.training.gradient_accumulation_steps}",
            "",
            "Loss Weights:",
            f"  Cross-Entropy: {self.loss.cross_entropy_weight}",
            f"  PV MSE: {self.loss.pv_mse_weight}",
            f"  PV CRPS: {self.loss.pv_crps_weight}",
            f"  Physics: {self.loss.moment_constraint_weight}",
            f"  Joint Training: {self.loss.joint_training}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic operations (may reduce performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Enable deterministic algorithms with warning for non-deterministic ops
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:
                # Older PyTorch versions
                pass
    
    # Set environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(config: ExperimentConfig) -> 'torch.device':
    """
    Get appropriate PyTorch device based on configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        PyTorch device object
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    
    if torch.cuda.is_available() and config.training.n_gpus > 0:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def count_parameters(model: 'torch.nn.Module') -> Dict[str, Union[int, float]]:
    """
    Count model parameters with detailed breakdown.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter statistics
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    # Count by module type
    by_module = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            module_params = sum(p.numel() for p in module.parameters(recurse=False))
            module_type = type(module).__name__
            by_module[module_type] = by_module.get(module_type, 0) + module_params
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
        "by_module_type": by_module,
    }


def get_lr(optimizer: 'torch.optim.Optimizer') -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


# ============================================================================
# Predefined Configurations for Ablation Studies
# ============================================================================

def get_ablation_configs() -> Dict[str, ExperimentConfig]:
    """
    Get predefined configurations for systematic ablation studies.
    
    Each ablation isolates a single component to assess its contribution
    to overall model performance.
    
    Returns:
        Dictionary mapping ablation names to configurations
    """
    configs = {}
    
    # Full model (baseline for comparison)
    full_model = ExperimentConfig(
        experiment_name="ablation_full_model",
        description="Full SkyGPT-ViTODE model with all components enabled"
    )
    configs["full_model"] = full_model
    
    # Ablation 1: Original SkyGPT baseline (no ViT, no Neural ODE)
    baseline = ExperimentConfig(
        experiment_name="ablation_original_skygpt",
        description="Original SkyGPT architecture (baseline without enhancements)"
    )
    baseline.vit.output_type = "deterministic"
    baseline.vit.depth = 0  # Indicates U-Net mode
    baseline.neural_ode.use_physics_constraint = True
    baseline.loss.joint_training = False
    configs["original_skygpt"] = baseline
    
    # Ablation 2: Without Neural ODE (discrete PhyCell only)
    no_ode = ExperimentConfig(
        experiment_name="ablation_no_neural_ode",
        description="Without Neural ODE continuous dynamics"
    )
    no_ode.neural_ode.solver = "euler"
    no_ode.neural_ode.integration_times = 1
    no_ode.loss.kinetic_energy_weight = 0.0
    configs["no_neural_ode"] = no_ode
    
    # Ablation 3: Without physics constraints
    no_physics = ExperimentConfig(
        experiment_name="ablation_no_physics",
        description="Without physics moment constraints"
    )
    no_physics.neural_ode.use_physics_constraint = False
    no_physics.loss.moment_constraint_weight = 0.0
    configs["no_physics"] = no_physics
    
    # Ablation 4: Without Vision Transformer (using U-Net)
    no_vit = ExperimentConfig(
        experiment_name="ablation_no_vit",
        description="Using U-Net instead of ViT for PV prediction"
    )
    no_vit.vit.depth = 0
    configs["no_vit"] = no_vit
    
    # Ablation 5: Without conformal prediction
    no_conformal = ExperimentConfig(
        experiment_name="ablation_no_conformal",
        description="Without conformal prediction calibration"
    )
    no_conformal.vit.output_type = "gaussian"
    configs["no_conformal"] = no_conformal
    
    # Ablation 6: Without joint training
    no_joint = ExperimentConfig(
        experiment_name="ablation_no_joint_training",
        description="Separate training without PV feedback to video generation"
    )
    no_joint.loss.joint_training = False
    no_joint.loss.pv_feedback_weight = 0.0
    configs["no_joint_training"] = no_joint
    
    # Ablation 7: Without MDN (Gaussian output)
    gaussian_output = ExperimentConfig(
        experiment_name="ablation_gaussian_output",
        description="Gaussian output instead of MDN"
    )
    gaussian_output.vit.output_type = "gaussian"
    configs["gaussian_output"] = gaussian_output
    
    # Ablation 8: Deterministic output
    deterministic = ExperimentConfig(
        experiment_name="ablation_deterministic",
        description="Deterministic output (no uncertainty quantification)"
    )
    deterministic.vit.output_type = "deterministic"
    deterministic.loss.pv_crps_weight = 0.0
    configs["deterministic"] = deterministic
    
    # Sensitivity studies: ViT sizes
    vit_tiny = ExperimentConfig(
        experiment_name="sensitivity_vit_tiny",
        description="Tiny ViT variant (reduced capacity)"
    )
    vit_tiny.vit.embed_dim = 192
    vit_tiny.vit.depth = 4
    vit_tiny.vit.n_heads = 3
    configs["vit_tiny"] = vit_tiny
    
    vit_small = ExperimentConfig(
        experiment_name="sensitivity_vit_small",
        description="Small ViT variant"
    )
    vit_small.vit.embed_dim = 256
    vit_small.vit.depth = 6
    vit_small.vit.n_heads = 4
    configs["vit_small"] = vit_small
    
    vit_large = ExperimentConfig(
        experiment_name="sensitivity_vit_large",
        description="Large ViT variant (increased capacity)"
    )
    vit_large.vit.embed_dim = 768
    vit_large.vit.depth = 12
    vit_large.vit.n_heads = 12
    configs["vit_large"] = vit_large
    
    # Sensitivity studies: MDN components
    for n_components in [3, 5, 7, 10]:
        mdn_config = ExperimentConfig(
            experiment_name=f"sensitivity_mdn_{n_components}",
            description=f"MDN with {n_components} mixture components"
        )
        mdn_config.vit.n_mixture_components = n_components
        configs[f"mdn_{n_components}"] = mdn_config
    
    # Sensitivity studies: ODE solvers
    for solver in ["euler", "rk4", "dopri5"]:
        solver_config = ExperimentConfig(
            experiment_name=f"sensitivity_solver_{solver}",
            description=f"Neural ODE with {solver} solver"
        )
        solver_config.neural_ode.solver = solver
        configs[f"solver_{solver}"] = solver_config
    
    return configs


def get_default_config() -> ExperimentConfig:
    """
    Get default configuration for quick experimentation.
    
    Returns:
        Default ExperimentConfig instance
    """
    return ExperimentConfig()


# ============================================================================
# Command-Line Interface Support
# ============================================================================

def parse_config_overrides(override_strings: List[str]) -> Dict[str, Any]:
    """
    Parse command-line configuration overrides.
    
    Supports format: "key=value" or "section.key=value"
    
    Args:
        override_strings: List of override strings
        
    Returns:
        Dictionary of parsed overrides
    """
    overrides = {}
    
    for override in override_strings:
        if '=' not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'")
        
        key, value_str = override.split('=', 1)
        key = key.strip()
        value_str = value_str.strip()
        
        # Parse value type
        if value_str.lower() == 'true':
            value = True
        elif value_str.lower() == 'false':
            value = False
        elif value_str.lower() == 'none':
            value = None
        else:
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str
        
        overrides[key] = value
    
    return overrides


# ============================================================================
# Main Entry Point for Testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SkyGPT-ViTODE Configuration Manager")
    parser.add_argument('--save', type=str, default=None, help='Save config to path')
    parser.add_argument('--load', type=str, default=None, help='Load config from path')
    parser.add_argument('--ablation', type=str, default=None, help='Show ablation config')
    parser.add_argument('--list-ablations', action='store_true', help='List all ablation configs')
    parser.add_argument('--override', nargs='*', default=[], help='Config overrides (key=value)')
    
    args = parser.parse_args()
    
    if args.list_ablations:
        print("Available Ablation Configurations:")
        print("-" * 50)
        for name, config in get_ablation_configs().items():
            print(f"  {name}: {config.description}")
        sys.exit(0)
    
    if args.ablation:
        ablations = get_ablation_configs()
        if args.ablation not in ablations:
            print(f"Unknown ablation: {args.ablation}")
            print(f"Available: {list(ablations.keys())}")
            sys.exit(1)
        config = ablations[args.ablation]
    elif args.load:
        config = ExperimentConfig.load(args.load)
    else:
        config = ExperimentConfig()
    
    # Apply overrides
    if args.override:
        overrides = parse_config_overrides(args.override)
        config = config.merge(overrides)
    
    # Display configuration
    print(config.summary())
    
    # Save if requested
    if args.save:
        saved_path = config.save(args.save)
        print(f"\nConfiguration saved to: {saved_path}")
    
    # Test round-trip serialization
    print("\n" + "=" * 70)
    print("Testing Configuration Serialization")
    print("=" * 70)
    
    # Save and reload
    test_path = config.save()
    loaded_config = ExperimentConfig.load(test_path)
    
    # Verify hash consistency
    original_hash = config.get_config_hash()
    loaded_hash = loaded_config.get_config_hash()
    
    print(f"Original hash:  {original_hash}")
    print(f"Loaded hash:    {loaded_hash}")
    print(f"Hash match:     {original_hash == loaded_hash}")
    
    # Check for differences
    diffs = config.diff(loaded_config)
    if diffs:
        print(f"\nDifferences found ({len(diffs)}):")
        for key, (orig, loaded) in diffs.items():
            print(f"  {key}: {orig} -> {loaded}")
    else:
        print("\nNo differences found. Serialization successful!")
    
    # Clean up test file
    os.remove(test_path)
    
    print("\n" + "=" * 70)
    print("Configuration module validated successfully!")
    print("=" * 70)