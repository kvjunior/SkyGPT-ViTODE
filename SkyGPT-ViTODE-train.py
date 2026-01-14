"""
SkyGPT-ViTODE Training Module
=============================

This module provides comprehensive training infrastructure for the SkyGPT-ViTODE
framework for probabilistic ultra-short-term solar forecasting.

Features:
    - PyTorch Lightning 2.0+ compatible training module
    - Multi-GPU distributed training with DDP strategy
    - Gradient accumulation for effective batch size scaling
    - Mixed precision training with configurable dtype (float16/bfloat16)
    - Comprehensive probabilistic metrics (CRPS, Winkler, reliability diagrams)
    - Conformal prediction calibration during training
    - Skill score computation against persistence and climatology baselines
    - Ablation study orchestration with automated result collection
    - Checkpoint management and experiment tracking
    - Learning rate warmup with configurable start learning rate
    - Model profiling and parameter counting utilities
    - Reliability diagram data collection for calibration analysis

Hardware Target:
    4× NVIDIA GeForce RTX 3090 (24GB each)
    Intel Xeon Silver 4314 CPU (64 cores)
    384GB DDR4 RAM
"""

from __future__ import annotations

import os
import sys
import gc
import math
import time
import json
import logging
import argparse
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Union, Callable
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau,
    OneCycleLR, LambdaLR, LinearLR, SequentialLR,
    ExponentialLR, MultiStepLR
)
from torch.cuda.amp import GradScaler

try:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer, seed_everything, LightningModule
    from pytorch_lightning.callbacks import (
        ModelCheckpoint, EarlyStopping, LearningRateMonitor,
        RichProgressBar, GradientAccumulationScheduler, Callback,
        StochasticWeightAveraging, BasePredictionWriter
    )
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
    from pytorch_lightning.strategies import DDPStrategy
    from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
    from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
    LIGHTNING_AVAILABLE = True
    LIGHTNING_VERSION = tuple(int(x) for x in pl.__version__.split('.')[:2])
except ImportError:
    LIGHTNING_AVAILABLE = False
    LIGHTNING_VERSION = (0, 0)
    pl = None

try:
    import torchmetrics
    from torchmetrics import Metric, MetricCollection
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    torchmetrics = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# Constants and Type Definitions
# ============================================================================

EPS = 1e-8
LOG_EPS = 1e-10
INF = float('inf')

# Gaussian quantiles for common coverage levels (two-tailed)
GAUSSIAN_QUANTILES = {
    0.50: 0.6745,
    0.60: 0.8416,
    0.70: 1.0364,
    0.80: 1.2816,
    0.85: 1.4395,
    0.90: 1.6449,
    0.95: 1.9600,
    0.99: 2.5758
}


@dataclass
class TrainingState:
    """Encapsulates training state for checkpointing and resumption."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = INF
    best_epoch: int = 0
    early_stop_counter: int = 0
    learning_rates: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'early_stop_counter': self.early_stop_counter,
            'learning_rates': self.learning_rates,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingState':
        """Create from dictionary."""
        return cls(**d)


# ============================================================================
# Logging Setup
# ============================================================================

logger = logging.getLogger('SkyGPT-ViTODE.Train')


def setup_training_logging(
    log_dir: str,
    experiment_name: str,
    level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """
    Setup training-specific logging with file and optional console handlers.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Experiment name for log file
        level: Logging level
        console: Whether to add console handler
        
    Returns:
        Configured logger
    """
    log_path = Path(log_dir) / experiment_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"train_{timestamp}.log"
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(console_handler)
    
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    
    return logger


@contextmanager
def log_timing(operation: str, log_fn: Callable = logger.info):
    """Context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log_fn(f"{operation}: {elapsed:.2f}s")


# ============================================================================
# Utility Functions
# ============================================================================

def compute_gaussian_quantile(coverage: float) -> float:
    """
    Compute Gaussian quantile for given coverage level.
    
    Args:
        coverage: Coverage probability (e.g., 0.90 for 90%)
        
    Returns:
        Quantile value for symmetric prediction interval
    """
    if coverage in GAUSSIAN_QUANTILES:
        return GAUSSIAN_QUANTILES[coverage]
    
    if SCIPY_AVAILABLE:
        return scipy_stats.norm.ppf((1 + coverage) / 2)
    
    # Fallback: linear interpolation between known quantiles
    coverages = sorted(GAUSSIAN_QUANTILES.keys())
    for i in range(len(coverages) - 1):
        if coverages[i] <= coverage <= coverages[i + 1]:
            t = (coverage - coverages[i]) / (coverages[i + 1] - coverages[i])
            q1 = GAUSSIAN_QUANTILES[coverages[i]]
            q2 = GAUSSIAN_QUANTILES[coverages[i + 1]]
            return q1 + t * (q2 - q1)
    
    # Default to 90% if out of range
    return 1.6449


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Count model parameters with detailed breakdown.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    # Breakdown by module type
    breakdown = defaultdict(int)
    for name, module in model.named_modules():
        module_params = sum(p.numel() for p in module.parameters(recurse=False))
        if module_params > 0:
            module_type = type(module).__name__
            breakdown[module_type] += module_params
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6,
        'frozen_millions': frozen / 1e6,
        'breakdown': dict(breakdown)
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information in GB."""
    if not torch.cuda.is_available():
        return {}
    
    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info[f'gpu_{i}'] = {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved
        }
    return info


# ============================================================================
# Custom Metrics (TorchMetrics Compatible)
# ============================================================================

if TORCHMETRICS_AVAILABLE:
    
    class CRPSMetric(Metric):
        """
        Continuous Ranked Probability Score (CRPS) metric.
        
        CRPS is a strictly proper scoring rule for probabilistic forecasts that
        measures both calibration and sharpness. It generalizes MAE to 
        probabilistic forecasts.
        
        For a Gaussian distribution, CRPS has a closed-form solution:
        CRPS(N(μ,σ), y) = σ[z(2Φ(z) - 1) + 2φ(z) - 1/√π]
        where z = (y - μ)/σ, Φ is CDF, φ is PDF
        
        Reference:
            Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
            prediction, and estimation. JASA, 102(477), 359-378.
        """
        
        full_state_update: bool = False
        higher_is_better: bool = False
        
        def __init__(
            self,
            n_samples: int = 100,
            use_closed_form: bool = True,
            **kwargs
        ):
            """
            Initialize CRPS metric.
            
            Args:
                n_samples: Number of samples for Monte Carlo estimation
                use_closed_form: Use closed-form for Gaussian (faster, more accurate)
            """
            super().__init__(**kwargs)
            self.n_samples = n_samples
            self.use_closed_form = use_closed_form
            
            self.add_state("crps_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
        def update(
            self,
            preds_mean: Tensor,
            preds_std: Tensor,
            targets: Tensor
        ):
            """
            Update metric with batch of predictions.
            
            Args:
                preds_mean: Predicted means (B,)
                preds_std: Predicted standard deviations (B,)
                targets: Target values (B,)
            """
            batch_size = targets.shape[0]
            
            if self.use_closed_form:
                # Closed-form CRPS for Gaussian
                crps = self._crps_gaussian(preds_mean, preds_std, targets)
            else:
                # Monte Carlo approximation
                crps = self._crps_monte_carlo(preds_mean, preds_std, targets)
            
            self.crps_sum += crps.sum()
            self.total += batch_size
        
        def _crps_gaussian(
            self,
            mean: Tensor,
            std: Tensor,
            target: Tensor
        ) -> Tensor:
            """
            Closed-form CRPS for Gaussian distribution.
            
            CRPS(N(μ,σ), y) = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
            """
            # Standardized error
            std_safe = std + EPS
            z = (target - mean) / std_safe
            
            # Standard normal CDF and PDF
            # Using approximation for numerical stability
            sqrt_2 = math.sqrt(2)
            sqrt_pi = math.sqrt(math.pi)
            
            # CDF: Φ(z) ≈ 0.5 * (1 + erf(z/√2))
            cdf = 0.5 * (1 + torch.erf(z / sqrt_2))
            
            # PDF: φ(z) = exp(-z²/2) / √(2π)
            pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
            
            # CRPS formula
            crps = std_safe * (z * (2 * cdf - 1) + 2 * pdf - 1 / sqrt_pi)
            
            return crps
        
        def _crps_monte_carlo(
            self,
            mean: Tensor,
            std: Tensor,
            target: Tensor
        ) -> Tensor:
            """Monte Carlo approximation of CRPS."""
            batch_size = target.shape[0]
            
            # Sample from predicted distribution
            samples = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                batch_size, self.n_samples, device=target.device
            )
            
            # CRPS = E|Y - y| - 0.5 * E|Y - Y'|
            target_exp = target.unsqueeze(1)
            term1 = torch.abs(samples - target_exp).mean(dim=1)
            
            # Pairwise term approximation using sorted samples
            samples_sorted, _ = torch.sort(samples, dim=1)
            n = self.n_samples
            weights = 2 * torch.arange(1, n + 1, device=samples.device).float() - n - 1
            term2 = (weights * samples_sorted).sum(dim=1) / (n * (n - 1))
            
            return term1 - 0.5 * torch.abs(term2)
        
        def compute(self) -> Tensor:
            """Compute final CRPS value."""
            return self.crps_sum / (self.total + EPS)
    
    
    class CoverageMetric(Metric):
        """
        Prediction Interval Coverage Probability (PICP) metric.
        
        Measures the empirical coverage of prediction intervals.
        Ideal: empirical coverage ≈ nominal coverage level.
        """
        
        full_state_update: bool = False
        higher_is_better: bool = None  # Depends on target coverage
        
        def __init__(self, coverage_level: float = 0.90, **kwargs):
            """
            Initialize coverage metric.
            
            Args:
                coverage_level: Nominal coverage level (e.g., 0.90)
            """
            super().__init__(**kwargs)
            self.coverage_level = coverage_level
            self.quantile = compute_gaussian_quantile(coverage_level)
            
            self.add_state("covered", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
        def update(
            self,
            preds_mean: Tensor,
            preds_std: Tensor,
            targets: Tensor
        ):
            """Update metric."""
            lower = preds_mean - self.quantile * preds_std
            upper = preds_mean + self.quantile * preds_std
            
            covered = ((targets >= lower) & (targets <= upper)).sum()
            
            self.covered += covered
            self.total += targets.numel()
        
        def compute(self) -> Tensor:
            """Compute empirical coverage."""
            return self.covered.float() / (self.total + EPS)
    
    
    class SharpnessMetric(Metric):
        """
        Sharpness metric (average prediction interval width).
        
        Measures the average width of prediction intervals.
        Lower is better, given proper coverage.
        """
        
        full_state_update: bool = False
        higher_is_better: bool = False
        
        def __init__(self, coverage_level: float = 0.90, **kwargs):
            """
            Initialize sharpness metric.
            
            Args:
                coverage_level: Coverage level for interval width
            """
            super().__init__(**kwargs)
            self.quantile = compute_gaussian_quantile(coverage_level)
            
            self.add_state("width_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
        def update(self, preds_std: Tensor):
            """Update metric."""
            width = 2 * self.quantile * preds_std
            self.width_sum += width.sum()
            self.total += preds_std.numel()
        
        def compute(self) -> Tensor:
            """Compute average interval width."""
            return self.width_sum / (self.total + EPS)
    
    
    class WinklerScoreMetric(Metric):
        """
        Winkler Score metric for prediction interval evaluation.
        
        Penalizes both interval width and coverage failures:
        - If y ∈ [L, U]: score = U - L
        - If y < L: score = (U - L) + (2/α)(L - y)
        - If y > U: score = (U - L) + (2/α)(y - U)
        
        Reference:
            Winkler, R. L. (1972). A Decision-Theoretic Approach to Interval 
            Estimation. JASA, 67(337), 187-191.
        """
        
        full_state_update: bool = False
        higher_is_better: bool = False
        
        def __init__(self, alpha: float = 0.10, **kwargs):
            """
            Initialize Winkler Score metric.
            
            Args:
                alpha: Significance level (1 - coverage probability)
            """
            super().__init__(**kwargs)
            self.alpha = alpha
            self.quantile = compute_gaussian_quantile(1 - alpha)
            
            self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
        def update(
            self,
            preds_mean: Tensor,
            preds_std: Tensor,
            targets: Tensor
        ):
            """Update metric."""
            lower = preds_mean - self.quantile * preds_std
            upper = preds_mean + self.quantile * preds_std
            delta = upper - lower
            
            below_penalty = (2 / self.alpha) * (lower - targets) * (targets < lower).float()
            above_penalty = (2 / self.alpha) * (targets - upper) * (targets > upper).float()
            
            score = delta + below_penalty + above_penalty
            
            self.score_sum += score.sum()
            self.total += targets.numel()
        
        def compute(self) -> Tensor:
            """Compute average Winkler score."""
            return self.score_sum / (self.total + EPS)
    
    
    class CalibrationMetric(Metric):
        """
        Calibration metric measuring deviation from ideal calibration.
        
        For a perfectly calibrated model, the empirical coverage at level p
        should equal p for all p ∈ [0, 1].
        
        Computes: √(mean((empirical_coverage_p - p)²)) over multiple p values.
        """
        
        full_state_update: bool = True
        higher_is_better: bool = False
        
        def __init__(
            self,
            n_bins: int = 10,
            **kwargs
        ):
            """
            Initialize calibration metric.
            
            Args:
                n_bins: Number of probability bins
            """
            super().__init__(**kwargs)
            self.n_bins = n_bins
            self.coverage_levels = torch.linspace(0.1, 0.9, n_bins)
            
            # Store predictions for calibration computation
            self.add_state("all_z_scores", default=[], dist_reduce_fx="cat")
        
        def update(
            self,
            preds_mean: Tensor,
            preds_std: Tensor,
            targets: Tensor
        ):
            """Update with batch."""
            z_scores = torch.abs((targets - preds_mean) / (preds_std + EPS))
            self.all_z_scores.append(z_scores)
        
        def compute(self) -> Tensor:
            """Compute calibration error."""
            if len(self.all_z_scores) == 0:
                return torch.tensor(0.0)
            
            z_scores = torch.cat(self.all_z_scores)
            device = z_scores.device
            
            calibration_errors = []
            for level in self.coverage_levels:
                quantile = compute_gaussian_quantile(level.item())
                empirical = (z_scores <= quantile).float().mean()
                error = (empirical - level.to(device)) ** 2
                calibration_errors.append(error)
            
            return torch.sqrt(torch.stack(calibration_errors).mean())
    
    
    class SkillScoreMetric(Metric):
        """
        Forecast Skill Score relative to a baseline.
        
        SS = 1 - MSE_forecast / MSE_baseline
        
        SS > 0: forecast better than baseline
        SS = 0: forecast equals baseline
        SS < 0: forecast worse than baseline
        """
        
        full_state_update: bool = False
        higher_is_better: bool = True
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            self.add_state("forecast_mse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("baseline_mse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
        def update(
            self,
            preds: Tensor,
            targets: Tensor,
            baseline_preds: Tensor
        ):
            """Update metric."""
            forecast_mse = ((preds - targets) ** 2).sum()
            baseline_mse = ((baseline_preds - targets) ** 2).sum()
            
            self.forecast_mse_sum += forecast_mse
            self.baseline_mse_sum += baseline_mse
            self.total += targets.numel()
        
        def compute(self) -> Tensor:
            """Compute skill score."""
            forecast_mse = self.forecast_mse_sum / (self.total + EPS)
            baseline_mse = self.baseline_mse_sum / (self.total + EPS)
            
            return 1 - forecast_mse / (baseline_mse + EPS)


# ============================================================================
# Loss Functions
# ============================================================================

class GaussianNLLLoss(nn.Module):
    """
    Gaussian negative log-likelihood loss with numerical stability.
    
    NLL = 0.5 * [log(2π) + log(σ²) + (y - μ)² / σ²]
        = 0.5 * [log(2π) + log_var + (y - μ)² * exp(-log_var)]
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        full: bool = False
    ):
        """
        Initialize Gaussian NLL loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
            full: Include constant log(2π) term
        """
        super().__init__()
        self.reduction = reduction
        self.full = full
        self.log_2pi = math.log(2 * math.pi)
    
    def forward(
        self,
        mean: Tensor,
        log_var: Tensor,
        target: Tensor
    ) -> Tensor:
        """
        Compute Gaussian NLL.
        
        Args:
            mean: Predicted mean (B,)
            log_var: Predicted log variance (B,)
            target: Target values (B,)
            
        Returns:
            NLL loss
        """
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-20, max=20)
        
        precision = torch.exp(-log_var)
        mse = (target - mean) ** 2
        
        nll = 0.5 * (log_var + mse * precision)
        
        if self.full:
            nll = nll + 0.5 * self.log_2pi
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        return nll


class HuberLoss(nn.Module):
    """Smooth L1 / Huber loss with configurable delta."""
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class QuantileLoss(nn.Module):
    """
    Quantile (Pinball) Loss for quantile regression.
    
    L_τ(y, q) = τ * max(y - q, 0) + (1 - τ) * max(q - y, 0)
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        reduction: str = 'mean'
    ):
        super().__init__()
        self.quantiles = quantiles
        self.reduction = reduction
    
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor
    ) -> Tensor:
        """
        Compute quantile loss.
        
        Args:
            predictions: Shape (B, num_quantiles)
            targets: Shape (B,)
        """
        targets = targets.unsqueeze(-1)
        errors = targets - predictions
        
        losses = []
        for i, tau in enumerate(self.quantiles):
            loss = torch.max(tau * errors[:, i], (tau - 1) * errors[:, i])
            losses.append(loss)
        
        total_loss = torch.stack(losses, dim=1).mean(dim=1)
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        return total_loss


# ============================================================================
# Baselines for Skill Score Computation
# ============================================================================

class PersistenceBaseline:
    """
    Persistence baseline for skill score computation.
    
    Predicts that future values equal the most recent observation.
    This is a strong baseline for ultra-short-term forecasting.
    """
    
    @staticmethod
    def predict(pv_history: Tensor) -> Tensor:
        """
        Return last observed value as prediction.
        
        Args:
            pv_history: Historical PV values (B, T)
            
        Returns:
            Persistence prediction (B,)
        """
        return pv_history[:, -1]
    
    @staticmethod
    def compute_error(pv_history: Tensor, pv_target: Tensor) -> Tensor:
        """
        Compute persistence baseline error.
        
        Args:
            pv_history: Historical PV values (B, T)
            pv_target: Target PV values (B,)
            
        Returns:
            Squared error per sample (B,)
        """
        persistence_pred = pv_history[:, -1]
        return (persistence_pred - pv_target) ** 2


class ClimatologyBaseline:
    """
    Climatology baseline for skill score computation.
    
    Predicts the historical mean as the forecast.
    Maintains running statistics for online computation.
    """
    
    def __init__(self):
        self.running_mean: Optional[Tensor] = None
        self.running_var: Optional[Tensor] = None
        self.count: int = 0
    
    def reset(self):
        """Reset running statistics."""
        self.running_mean = None
        self.running_var = None
        self.count = 0
    
    def update(self, values: Tensor):
        """
        Update running mean with Welford's online algorithm.
        
        Args:
            values: New values to incorporate
        """
        values = values.flatten()
        n = values.numel()
        
        if self.running_mean is None:
            self.running_mean = values.mean()
            self.running_var = values.var() if n > 1 else torch.zeros_like(self.running_mean)
            self.count = n
        else:
            # Welford's online algorithm for numerical stability
            old_count = self.count
            self.count = old_count + n
            
            batch_mean = values.mean()
            delta = batch_mean - self.running_mean
            self.running_mean = self.running_mean + delta * n / self.count
            
            if n > 1:
                batch_var = values.var()
                m2_batch = batch_var * (n - 1)
                m2_old = self.running_var * (old_count - 1) if old_count > 1 else 0
                m2 = m2_old + m2_batch + delta ** 2 * old_count * n / self.count
                self.running_var = m2 / (self.count - 1)
    
    def predict(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Return climatology prediction.
        
        Args:
            batch_size: Number of predictions
            device: Device for output tensor
            
        Returns:
            Climatology predictions (batch_size,)
        """
        if self.running_mean is None:
            return torch.zeros(batch_size, device=device)
        return torch.full((batch_size,), self.running_mean.item(), device=device)
    
    def compute_error(self, pv_target: Tensor) -> Tensor:
        """
        Compute climatology baseline error.
        
        Args:
            pv_target: Target PV values (B,)
            
        Returns:
            Squared error per sample (B,)
        """
        if self.running_mean is None:
            return torch.zeros_like(pv_target)
        return (self.running_mean - pv_target) ** 2


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

def create_scheduler(
    optimizer: optim.Optimizer,
    config,
    total_steps: int,
    steps_per_epoch: int
) -> Tuple[Any, str]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer instance
        config: TrainingConfig object
        total_steps: Total training steps
        steps_per_epoch: Steps per epoch
        
    Returns:
        Tuple of (scheduler, interval) where interval is 'step' or 'epoch'
    """
    scheduler_type = config.scheduler
    
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config.min_lr
        )
        interval = "step"
        
    elif scheduler_type == "step":
        # Step every 30 epochs by default
        step_size = getattr(config, 'lr_step_size', 30) * steps_per_epoch
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=getattr(config, 'lr_gamma', 0.5)
        )
        interval = "step"
        
    elif scheduler_type == "multistep":
        # Milestones at 60% and 80% of training
        milestones = [
            int(0.6 * config.max_epochs) * steps_per_epoch,
            int(0.8 * config.max_epochs) * steps_per_epoch
        ]
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=getattr(config, 'lr_gamma', 0.1)
        )
        interval = "step"
        
    elif scheduler_type == "exponential":
        # Decay to min_lr over training
        gamma = (config.min_lr / config.learning_rate) ** (1 / config.max_epochs)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        interval = "epoch"
        
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=config.min_lr,
            verbose=True
        )
        interval = "epoch"
        
    elif scheduler_type == "warmup_cosine":
        warmup_steps = int(config.warmup_epochs * steps_per_epoch)
        
        # Warmup from warmup_start_lr to learning_rate
        warmup_start_factor = getattr(config, 'warmup_start_lr', 1e-7) / config.learning_rate
        warmup_start_factor = max(warmup_start_factor, 1e-8)  # Prevent zero
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Cosine decay after warmup
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=config.min_lr
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        interval = "step"
        
    elif scheduler_type == "linear":
        # Linear decay from learning_rate to min_lr
        end_factor = config.min_lr / config.learning_rate
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=max(end_factor, 1e-8),
            total_iters=total_steps
        )
        interval = "step"
        
    elif scheduler_type == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_epochs / config.max_epochs,
            anneal_strategy='cos',
            final_div_factor=config.learning_rate / max(config.min_lr, 1e-8)
        )
        interval = "step"
        
    elif scheduler_type == "constant":
        # No scheduling
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        interval = "step"
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler, interval


# ============================================================================
# Custom Callbacks
# ============================================================================

class ConformalCalibrationCallback(Callback):
    """
    Callback to perform conformal calibration on validation set.
    
    Calibrates the conformal predictor periodically during training
    to ensure well-calibrated prediction intervals.
    """
    
    def __init__(
        self,
        calibration_frequency: int = 5,
        calibration_fraction: float = 0.5
    ):
        """
        Initialize callback.
        
        Args:
            calibration_frequency: Calibrate every N epochs
            calibration_fraction: Fraction of validation set for calibration
        """
        super().__init__()
        self.calibration_frequency = calibration_frequency
        self.calibration_fraction = calibration_fraction
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Calibrate conformal predictor after validation."""
        if trainer.current_epoch % self.calibration_frequency != 0:
            return
        
        if not hasattr(pl_module.model, 'conformal'):
            return
        
        if pl_module.model.conformal is None:
            return
        
        # Collect validation predictions
        predictions_list = []
        uncertainties_list = []
        targets_list = []
        
        n_batches = 0
        max_batches = int(len(trainer.val_dataloaders[0]) * self.calibration_fraction)
        
        pl_module.eval()
        with torch.no_grad():
            for batch in trainer.val_dataloaders[0]:
                if n_batches >= max_batches:
                    break
                
                # Move batch to device
                batch = {
                    k: v.to(pl_module.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass for PV prediction
                pv_output = pl_module.model.predict_pv(batch['video'])
                
                pred_mean = pv_output.get('mean', pv_output.get('prediction'))
                pred_std = pv_output.get('std', torch.zeros_like(pred_mean))
                
                predictions_list.append(pred_mean.cpu())
                uncertainties_list.append(pred_std.cpu())
                targets_list.append(batch['pv_target'].cpu())
                
                n_batches += 1
        
        # Concatenate and calibrate
        if predictions_list:
            predictions = torch.cat(predictions_list)
            uncertainties = torch.cat(uncertainties_list)
            targets = torch.cat(targets_list)
            
            pl_module.model.conformal.calibrate(
                predictions.to(pl_module.device),
                uncertainties.to(pl_module.device),
                targets.to(pl_module.device)
            )
            
            if hasattr(pl_module.model.conformal, 'calibrated_quantile'):
                quantile = pl_module.model.conformal.calibrated_quantile
                logger.info(
                    f"Epoch {trainer.current_epoch}: Conformal predictor calibrated "
                    f"(quantile={quantile:.4f})"
                )


class MetricLoggingCallback(Callback):
    """
    Callback for detailed metric logging at epoch end.
    """
    
    def __init__(self, log_gpu_memory: bool = True):
        super().__init__()
        self.log_gpu_memory = log_gpu_memory
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Log training epoch summary."""
        metrics = trainer.callback_metrics
        
        # Filter training metrics
        train_metrics = {
            k: v for k, v in metrics.items()
            if k.startswith('train/') and not k.startswith('train/epoch')
        }
        
        if train_metrics:
            metric_str = " | ".join([
                f"{k.split('/')[-1]}: {v:.4f}"
                for k, v in train_metrics.items()
            ])
            logger.info(f"Epoch {trainer.current_epoch} Train | {metric_str}")
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation epoch summary."""
        metrics = trainer.callback_metrics
        
        # Filter validation metrics
        val_metrics = {
            k: v for k, v in metrics.items()
            if k.startswith('val/') and isinstance(v, (int, float, Tensor))
        }
        
        if val_metrics:
            metric_str = " | ".join([
                f"{k.split('/')[-1]}: {v:.4f}" if isinstance(v, (float, Tensor)) else f"{k.split('/')[-1]}: {v}"
                for k, v in val_metrics.items()
            ])
            logger.info(f"Epoch {trainer.current_epoch} Val   | {metric_str}")
        
        # Log GPU memory
        if self.log_gpu_memory and torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                mem_str = " | ".join([
                    f"GPU{k.split('_')[1]}: {v['allocated_gb']:.1f}/{v['total_gb']:.1f}GB"
                    for k, v in gpu_info.items()
                ])
                logger.debug(f"Memory | {mem_str}")


class ReliabilityDiagramCallback(Callback):
    """
    Callback to collect data for reliability diagrams.
    
    Stores prediction-target pairs for post-hoc calibration analysis.
    """
    
    def __init__(
        self,
        save_dir: str,
        n_bins: int = 10,
        save_frequency: int = 10
    ):
        """
        Initialize callback.
        
        Args:
            save_dir: Directory to save reliability data
            n_bins: Number of bins for reliability diagram
            save_frequency: Save data every N epochs
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.n_bins = n_bins
        self.save_frequency = save_frequency
        
        # Storage for current epoch
        self.epoch_data = []
    
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx
    ):
        """Collect prediction data."""
        if trainer.current_epoch % self.save_frequency != 0:
            return
        
        if not hasattr(pl_module, '_last_pv_output'):
            return
        
        pv_output = pl_module._last_pv_output
        pv_target = batch['pv_target']
        
        pred_mean = pv_output.get('mean', pv_output.get('prediction'))
        pred_std = pv_output.get('std', torch.zeros_like(pred_mean))
        
        self.epoch_data.append({
            'pred_mean': pred_mean.detach().cpu().numpy(),
            'pred_std': pred_std.detach().cpu().numpy(),
            'target': pv_target.detach().cpu().numpy()
        })
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Save reliability data."""
        if trainer.current_epoch % self.save_frequency != 0:
            return
        
        if not self.epoch_data:
            return
        
        # Concatenate all data
        pred_mean = np.concatenate([d['pred_mean'] for d in self.epoch_data])
        pred_std = np.concatenate([d['pred_std'] for d in self.epoch_data])
        target = np.concatenate([d['target'] for d in self.epoch_data])
        
        # Compute reliability diagram data
        z_scores = np.abs((target - pred_mean) / (pred_std + 1e-8))
        
        reliability_data = {
            'epoch': trainer.current_epoch,
            'coverage_levels': [],
            'empirical_coverage': [],
            'n_samples': len(target)
        }
        
        for level in np.linspace(0.1, 0.99, 20):
            quantile = compute_gaussian_quantile(level)
            empirical = (z_scores <= quantile).mean()
            reliability_data['coverage_levels'].append(float(level))
            reliability_data['empirical_coverage'].append(float(empirical))
        
        # Save
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / f"reliability_epoch_{trainer.current_epoch:04d}.json"
        
        with open(save_path, 'w') as f:
            json.dump(reliability_data, f, indent=2)
        
        # Clear epoch data
        self.epoch_data.clear()


class GradientMonitorCallback(Callback):
    """
    Callback to monitor gradient statistics.
    
    Useful for debugging training issues like vanishing/exploding gradients.
    """
    
    def __init__(self, log_frequency: int = 100):
        """
        Initialize callback.
        
        Args:
            log_frequency: Log gradient stats every N steps
        """
        super().__init__()
        self.log_frequency = log_frequency
    
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Log gradient statistics before optimizer step."""
        if trainer.global_step % self.log_frequency != 0:
            return
        
        grad_norms = []
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
        
        if grad_norms:
            pl_module.log('debug/grad_norm_mean', np.mean(grad_norms))
            pl_module.log('debug/grad_norm_max', max(grad_norms))
            pl_module.log('debug/grad_norm_min', min(grad_norms))


# ============================================================================
# PyTorch Lightning Training Module
# ============================================================================

if LIGHTNING_AVAILABLE:
    
    class SkyGPTViTODELightning(pl.LightningModule):
        """
        PyTorch Lightning module for SkyGPT-ViTODE training.
        
        Provides comprehensive training with:
        - Multi-GPU distributed training
        - Mixed precision support (float16/bfloat16)
        - Gradient accumulation
        - Probabilistic metrics (CRPS, coverage, sharpness, Winkler)
        - Skill score computation against persistence and climatology baselines
        - Conformal prediction integration
        - Reliability diagram data collection
        
        Attributes:
            config: Experiment configuration
            model: SkyGPT-ViTODE model
            persistence: Persistence baseline
            climatology: Climatology baseline
            val_metrics: Validation metric collection
        """
        
        def __init__(self, config):
            """
            Initialize Lightning module.
            
            Args:
                config: ExperimentConfig object
            """
            super().__init__()
            self.config = config
            
            # Save hyperparameters
            self.save_hyperparameters(ignore=['config'])
            
            # Store config in hparams for checkpoint loading
            self.hparams['config_dict'] = config.to_dict() if hasattr(config, 'to_dict') else {}
            
            # Import and create model
            from models import create_model, count_parameters
            self.model = create_model(config)
            
            # Log model info
            params = count_parameters(self.model)
            logger.info(
                f"Model initialized: {params['total_millions']:.2f}M total, "
                f"{params['trainable_millions']:.2f}M trainable"
            )
            
            # Loss components
            self.gaussian_nll = GaussianNLLLoss()
            
            # Baselines for skill score
            self.persistence = PersistenceBaseline()
            self.climatology = ClimatologyBaseline()
            
            # Setup metrics
            self._setup_metrics()
            
            # Effective batch size for logging
            self.effective_batch_size = (
                config.data.batch_size *
                config.training.gradient_accumulation_steps *
                max(1, config.training.n_gpus)
            )
            
            # Store last PV output for reliability callback
            self._last_pv_output = None
        
        def _setup_metrics(self):
            """Initialize evaluation metrics."""
            if not TORCHMETRICS_AVAILABLE:
                logger.warning("TorchMetrics not available. Using basic metrics.")
                self.val_metrics = None
                return
            
            # Validation metrics
            coverage_levels = self.config.conformal.coverage_levels
            
            self.val_metrics = MetricCollection({
                'mae': torchmetrics.MeanAbsoluteError(),
                'mse': torchmetrics.MeanSquaredError(),
                'rmse': torchmetrics.MeanSquaredError(squared=False),
            })
            
            # Probabilistic metrics
            if self.config.vit.output_type in ['mdn', 'gaussian']:
                self.val_crps = CRPSMetric()
                self.val_coverage = nn.ModuleDict({
                    f"cov_{int(level*100)}": CoverageMetric(level)
                    for level in coverage_levels
                })
                self.val_sharpness = SharpnessMetric(coverage_levels[0])
                self.val_winkler = WinklerScoreMetric(alpha=1 - coverage_levels[0])
                self.val_calibration = CalibrationMetric()
            else:
                self.val_crps = None
                self.val_coverage = None
                self.val_sharpness = None
                self.val_winkler = None
                self.val_calibration = None
            
            # Skill score metrics
            if self.config.evaluation.compare_persistence:
                self.skill_persistence = SkillScoreMetric()
            else:
                self.skill_persistence = None
            
            if self.config.evaluation.compare_climatology:
                self.skill_climatology = SkillScoreMetric()
            else:
                self.skill_climatology = None
        
        def forward(
            self,
            video: Tensor,
            pv_target: Optional[Tensor] = None
        ) -> Dict[str, Tensor]:
            """Forward pass through model."""
            return self.model(video, pv_target)
        
        def _extract_predictions(
            self,
            pv_output: Dict[str, Tensor]
        ) -> Tuple[Tensor, Tensor]:
            """
            Extract mean and std from model output.
            
            Args:
                pv_output: Model output dictionary
                
            Returns:
                Tuple of (predicted mean, predicted std)
            """
            output_type = self.config.vit.output_type
            
            if output_type == "mdn":
                # Use model's built-in methods if available
                if 'mean' in pv_output and 'std' in pv_output:
                    pred_mean = pv_output['mean']
                    pred_std = pv_output['std']
                else:
                    weights = pv_output['weights']
                    means = pv_output['means']
                    scales = pv_output['scales']
                    
                    # Mixture mean
                    pred_mean = (weights * means).sum(dim=-1)
                    
                    # Mixture std (law of total variance)
                    var_within = (weights * scales ** 2).sum(dim=-1)
                    var_between = (weights * (means - pred_mean.unsqueeze(-1)) ** 2).sum(dim=-1)
                    pred_std = torch.sqrt(var_within + var_between + EPS)
                    
            elif output_type == "gaussian":
                pred_mean = pv_output['mean']
                pred_std = pv_output['std']
                
            else:  # deterministic
                pred_mean = pv_output.get('prediction', pv_output.get('mean'))
                pred_std = torch.zeros_like(pred_mean)
            
            return pred_mean, pred_std
        
        def _denormalize_pv(self, normalized: Tensor) -> Tensor:
            """
            Convert normalized PV values to physical units (kW).
            
            Args:
                normalized: Normalized values in [0, 1]
                
            Returns:
                Values in kW
            """
            pv_max = self.config.data.pv_max
            pv_min = getattr(self.config.data, 'pv_min', 0.0)
            return normalized * (pv_max - pv_min) + pv_min
        
        def training_step(
            self,
            batch: Dict[str, Tensor],
            batch_idx: int
        ) -> Tensor:
            """
            Execute training step.
            
            Args:
                batch: Data batch containing 'video', 'pv_target', 'pv_history'
                batch_idx: Batch index
                
            Returns:
                Training loss
            """
            video = batch['video']
            pv_target = batch['pv_target']
            
            # Forward pass
            output = self.model(video, pv_target)
            
            # Total loss from model
            loss = output['loss']
            
            # Batch size for proper averaging
            batch_size = video.shape[0]
            
            # Log losses with batch_size for proper weighted averaging
            self.log('train/loss', loss, on_step=True, on_epoch=True,
                    prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log('train/video_loss', output.get('video_loss', 0.0),
                    on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log('train/physics_loss', output.get('physics_loss', 0.0),
                    on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log('train/pv_loss', output.get('pv_loss', 0.0),
                    on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            
            # Kinetic energy regularization
            if 'kinetic_loss' in output:
                self.log('train/kinetic_loss', output['kinetic_loss'],
                        on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            
            # Log learning rate
            if batch_idx % 100 == 0:
                lr = self.optimizers().param_groups[0]['lr']
                self.log('train/lr', lr, on_step=True, on_epoch=False)
            
            return loss
        
        def validation_step(
            self,
            batch: Dict[str, Tensor],
            batch_idx: int
        ) -> Dict[str, Tensor]:
            """
            Execute validation step.
            
            Args:
                batch: Data batch
                batch_idx: Batch index
                
            Returns:
                Validation outputs
            """
            video = batch['video']
            pv_target = batch['pv_target']
            pv_history = batch.get('pv_history')
            batch_size = video.shape[0]
            
            # Forward pass
            output = self.model(video, pv_target)
            
            # Extract predictions
            pv_output = output['pv_output']
            self._last_pv_output = pv_output  # Store for reliability callback
            
            pred_mean, pred_std = self._extract_predictions(pv_output)
            
            # Denormalize for metrics in physical units
            pred_mean_kw = self._denormalize_pv(pred_mean)
            pv_range = self.config.data.pv_max - getattr(self.config.data, 'pv_min', 0.0)
            pred_std_kw = pred_std * pv_range
            target_kw = self._denormalize_pv(pv_target)
            
            # Log validation loss
            self.log('val/loss', output['loss'], on_step=False, on_epoch=True,
                    prog_bar=True, sync_dist=True, batch_size=batch_size)
            
            # Point prediction metrics
            if self.val_metrics is not None:
                self.val_metrics.update(pred_mean_kw, target_kw)
            
            # Probabilistic metrics
            if self.val_crps is not None:
                self.val_crps.update(pred_mean_kw, pred_std_kw, target_kw)
                
                for name, metric in self.val_coverage.items():
                    metric.update(pred_mean_kw, pred_std_kw, target_kw)
                
                self.val_sharpness.update(pred_std_kw)
                self.val_winkler.update(pred_mean_kw, pred_std_kw, target_kw)
                self.val_calibration.update(pred_mean_kw, pred_std_kw, target_kw)
            
            # Skill scores
            if pv_history is not None:
                pv_history_kw = self._denormalize_pv(pv_history)
                
                if self.skill_persistence is not None:
                    persistence_pred = self.persistence.predict(pv_history_kw)
                    self.skill_persistence.update(pred_mean_kw, target_kw, persistence_pred)
                
                if self.skill_climatology is not None:
                    self.climatology.update(target_kw)
                    climatology_pred = self.climatology.predict(batch_size, target_kw.device)
                    self.skill_climatology.update(pred_mean_kw, target_kw, climatology_pred)
            
            return {'val_loss': output['loss']}
        
        def on_validation_epoch_end(self):
            """Process validation epoch metrics."""
            # Point prediction metrics
            if self.val_metrics is not None:
                metrics = self.val_metrics.compute()
                for name, value in metrics.items():
                    self.log(f'val/{name}_kw', value, sync_dist=True)
                self.val_metrics.reset()
            
            # Probabilistic metrics
            if self.val_crps is not None:
                self.log('val/crps_kw', self.val_crps.compute(), sync_dist=True)
                self.val_crps.reset()
                
                for name, metric in self.val_coverage.items():
                    coverage_level = int(name.split('_')[1]) / 100
                    computed = metric.compute()
                    self.log(f'val/{name}', computed, sync_dist=True)
                    
                    # Log coverage deviation
                    deviation = torch.abs(computed - coverage_level)
                    self.log(f'val/{name}_deviation', deviation, sync_dist=True)
                    metric.reset()
                
                self.log('val/sharpness_kw', self.val_sharpness.compute(), sync_dist=True)
                self.val_sharpness.reset()
                
                self.log('val/winkler_kw', self.val_winkler.compute(), sync_dist=True)
                self.val_winkler.reset()
                
                self.log('val/calibration_error', self.val_calibration.compute(), sync_dist=True)
                self.val_calibration.reset()
            
            # Skill scores
            if self.skill_persistence is not None:
                self.log('val/skill_persistence', self.skill_persistence.compute(), sync_dist=True)
                self.skill_persistence.reset()
            
            if self.skill_climatology is not None:
                self.log('val/skill_climatology', self.skill_climatology.compute(), sync_dist=True)
                self.skill_climatology.reset()
        
        def test_step(
            self,
            batch: Dict[str, Tensor],
            batch_idx: int
        ) -> Dict[str, Tensor]:
            """
            Execute test step.
            
            Same as validation but may include additional metrics.
            """
            return self.validation_step(batch, batch_idx)
        
        def on_test_epoch_end(self):
            """Process test epoch metrics."""
            self.on_validation_epoch_end()
        
        def configure_optimizers(self):
            """
            Configure optimizer and learning rate scheduler.
            
            Returns:
                Dictionary with optimizer and scheduler configuration
            """
            config = self.config.training
            
            # Separate weight decay for different parameter groups
            decay_params = []
            no_decay_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # No weight decay for biases and normalization layers
                if 'bias' in name or 'norm' in name.lower() or 'bn' in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            
            param_groups = [
                {'params': decay_params, 'weight_decay': config.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            
            # Log parameter group sizes
            logger.info(
                f"Optimizer groups: {len(decay_params)} params with weight decay, "
                f"{len(no_decay_params)} params without"
            )
            
            # Create optimizer
            betas = tuple(config.betas) if hasattr(config.betas, '__iter__') else (0.9, 0.999)
            
            if config.optimizer == "adamw":
                optimizer = optim.AdamW(
                    param_groups,
                    lr=config.learning_rate,
                    betas=betas,
                    eps=config.eps
                )
            elif config.optimizer == "adam":
                optimizer = optim.Adam(
                    param_groups,
                    lr=config.learning_rate,
                    betas=betas,
                    eps=config.eps
                )
            elif config.optimizer == "sgd":
                optimizer = optim.SGD(
                    param_groups,
                    lr=config.learning_rate,
                    momentum=0.9,
                    nesterov=True
                )
            elif config.optimizer == "rmsprop":
                optimizer = optim.RMSprop(
                    param_groups,
                    lr=config.learning_rate,
                    alpha=0.99,
                    eps=config.eps
                )
            else:
                raise ValueError(f"Unknown optimizer: {config.optimizer}")
            
            # Compute total steps
            if config.max_steps is not None and config.max_steps > 0:
                total_steps = config.max_steps
            else:
                # Estimate from trainer
                if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches'):
                    total_steps = self.trainer.estimated_stepping_batches
                else:
                    # Fallback estimation
                    estimated_samples = 10000  # Conservative default
                    steps_per_epoch = max(1, estimated_samples // self.effective_batch_size)
                    total_steps = config.max_epochs * steps_per_epoch
            
            steps_per_epoch = max(1, total_steps // config.max_epochs)
            
            # Create scheduler
            scheduler, interval = create_scheduler(
                optimizer, config, total_steps, steps_per_epoch
            )
            
            scheduler_config = {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1,
                "name": "learning_rate"
            }
            
            # ReduceLROnPlateau needs monitor
            if config.scheduler == "plateau":
                scheduler_config["monitor"] = config.monitor_metric
                scheduler_config["strict"] = True
            
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        
        def lr_scheduler_step(self, scheduler, metric=None):
            """Custom scheduler step for ReduceLROnPlateau compatibility."""
            if isinstance(scheduler, ReduceLROnPlateau):
                if metric is not None:
                    scheduler.step(metric)
            else:
                scheduler.step()
        
        def on_train_start(self):
            """Log training configuration at start."""
            logger.info(f"Starting training with effective batch size: {self.effective_batch_size}")
            logger.info(f"Total epochs: {self.config.training.max_epochs}")
            
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                logger.info(f"Estimated total steps: {self.trainer.estimated_stepping_batches}")


# ============================================================================
# VQ-VAE Pre-training Module
# ============================================================================

if LIGHTNING_AVAILABLE:
    
    class VQVAELightning(pl.LightningModule):
        """
        PyTorch Lightning module for VQ-VAE pre-training.
        
        Trains the VQ-VAE component separately before full model training.
        Monitors reconstruction quality and codebook utilization.
        """
        
        def __init__(self, config):
            """
            Initialize VQ-VAE module.
            
            Args:
                config: ExperimentConfig object
            """
            super().__init__()
            self.config = config
            self.save_hyperparameters(ignore=['config'])
            
            # Create VQ-VAE
            from models import create_vqvae
            self.vqvae = create_vqvae(config)
            
            # Track codebook utilization
            self.codebook_usage = defaultdict(int)
        
        def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict]:
            """Forward pass."""
            return self.vqvae(x)
        
        def training_step(
            self,
            batch: Dict[str, Tensor],
            batch_idx: int
        ) -> Tensor:
            """Training step."""
            video = batch['video']
            batch_size = video.shape[0]
            
            recon_loss, recon, vq_output = self.vqvae(video)
            commitment_loss = vq_output['commitment_loss']
            
            total_loss = recon_loss + self.config.vqvae.commitment_cost * commitment_loss
            
            # Logging
            self.log('train/recon_loss', recon_loss, prog_bar=True, 
                    sync_dist=True, batch_size=batch_size)
            self.log('train/commitment_loss', commitment_loss, 
                    sync_dist=True, batch_size=batch_size)
            self.log('train/perplexity', vq_output['perplexity'], 
                    sync_dist=True, batch_size=batch_size)
            self.log('train/total_loss', total_loss, 
                    sync_dist=True, batch_size=batch_size)
            
            return total_loss
        
        def validation_step(
            self,
            batch: Dict[str, Tensor],
            batch_idx: int
        ) -> Dict:
            """Validation step."""
            video = batch['video']
            batch_size = video.shape[0]
            
            recon_loss, recon, vq_output = self.vqvae(video)
            
            self.log('val/recon_loss', recon_loss, prog_bar=True, 
                    sync_dist=True, batch_size=batch_size)
            self.log('val/perplexity', vq_output['perplexity'], 
                    sync_dist=True, batch_size=batch_size)
            
            # Track codebook utilization
            if 'encoding_indices' in vq_output:
                indices = vq_output['encoding_indices'].flatten()
                unique_codes = indices.unique().numel()
                utilization = unique_codes / self.config.vqvae.n_codes
                self.log('val/codebook_utilization', utilization, 
                        sync_dist=True, batch_size=batch_size)
            
            # PSNR for reconstruction quality
            mse = F.mse_loss(recon, video)
            psnr = 10 * torch.log10(1.0 / (mse + EPS))
            self.log('val/psnr', psnr, sync_dist=True, batch_size=batch_size)
            
            return {'val_loss': recon_loss}
        
        def configure_optimizers(self):
            """Configure optimizer."""
            optimizer = optim.AdamW(
                self.parameters(),
                lr=3e-4,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 50,
                eta_min=1e-6
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }


# ============================================================================
# Training Functions
# ============================================================================

def create_callbacks(config) -> List[Callback]:
    """
    Create training callbacks based on configuration.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_dir = Path(config.training.checkpoint_dir) / config.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='{epoch:03d}-{val/loss:.4f}',
        save_top_k=config.training.save_top_k,
        monitor=config.training.monitor_metric,
        mode=config.training.monitor_mode,
        save_last=config.training.save_last,
        every_n_epochs=config.training.save_every_n_epochs,
        auto_insert_metric_name=False,
        save_weights_only=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config.training.early_stopping:
        early_stop = EarlyStopping(
            monitor=config.training.monitor_metric,
            patience=config.training.patience,
            mode=config.training.monitor_mode,
            min_delta=config.training.min_delta,
            verbose=True,
            strict=True
        )
        callbacks.append(early_stop)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Progress bar
    callbacks.append(RichProgressBar())
    
    # Conformal calibration
    if hasattr(config, 'conformal') and config.conformal.use_adaptive:
        callbacks.append(ConformalCalibrationCallback(
            calibration_frequency=5,
            calibration_fraction=0.5
        ))
    
    # Custom metric logging
    callbacks.append(MetricLoggingCallback(log_gpu_memory=True))
    
    # Reliability diagram data collection
    if hasattr(config, 'evaluation') and getattr(config.evaluation, 'compute_reliability', False):
        reliability_dir = Path(config.results_dir) / config.experiment_name / 'reliability'
        callbacks.append(ReliabilityDiagramCallback(
            save_dir=str(reliability_dir),
            save_frequency=10
        ))
    
    # Stochastic Weight Averaging (optional)
    if getattr(config.training, 'use_swa', False):
        swa = StochasticWeightAveraging(
            swa_lrs=config.training.min_lr,
            swa_epoch_start=int(0.75 * config.training.max_epochs)
        )
        callbacks.append(swa)
    
    return callbacks


def create_loggers(config) -> List:
    """
    Create experiment loggers.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        List of loggers
    """
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=config.experiment_name,
        version=config.get_config_hash()[:8] if hasattr(config, 'get_config_hash') else None,
        default_hp_metric=False
    )
    loggers.append(tb_logger)
    
    # CSV logger for easy analysis
    csv_logger = CSVLogger(
        save_dir=str(log_dir),
        name=f"{config.experiment_name}_csv",
        version=config.get_config_hash()[:8] if hasattr(config, 'get_config_hash') else None
    )
    loggers.append(csv_logger)
    
    # Weights & Biases logger (optional)
    if getattr(config, 'use_wandb', False):
        try:
            wandb_logger = WandbLogger(
                project=getattr(config, 'wandb_project', 'skygpt-vitode'),
                name=config.experiment_name,
                save_dir=str(log_dir)
            )
            loggers.append(wandb_logger)
        except Exception as e:
            logger.warning(f"Failed to initialize W&B logger: {e}")
    
    return loggers


def create_trainer(
    config,
    callbacks: List,
    loggers: List,
    profiler: Optional[str] = None
) -> Trainer:
    """
    Create PyTorch Lightning trainer.
    
    Args:
        config: ExperimentConfig object
        callbacks: List of callbacks
        loggers: List of loggers
        profiler: Profiler type ('simple', 'advanced', or None)
        
    Returns:
        Configured Trainer
    """
    training_config = config.training
    
    # Determine accelerator and devices
    if torch.cuda.is_available() and training_config.n_gpus > 0:
        accelerator = "gpu"
        devices = training_config.n_gpus
    else:
        accelerator = "cpu"
        devices = 1
    
    # Strategy for multi-GPU
    if devices > 1:
        strategy = DDPStrategy(
            find_unused_parameters=getattr(training_config, 'find_unused_parameters', True),
            gradient_as_bucket_view=True,
            static_graph=False
        )
    else:
        strategy = "auto"
    
    # Precision setting with amp_dtype support
    if training_config.use_amp:
        amp_dtype = getattr(training_config, 'amp_dtype', 'float16')
        if amp_dtype == "bfloat16":
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    else:
        precision = "32-true"
    
    # Profiler
    profiler_instance = None
    if profiler == 'simple':
        profiler_instance = SimpleProfiler(
            dirpath=config.log_dir,
            filename='profile'
        )
    elif profiler == 'advanced':
        profiler_instance = AdvancedProfiler(
            dirpath=config.log_dir,
            filename='advanced_profile'
        )
    
    trainer = Trainer(
        max_epochs=training_config.max_epochs,
        max_steps=training_config.max_steps or -1,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=training_config.gradient_clip_val,
        gradient_clip_algorithm=training_config.gradient_clip_algorithm,
        accumulate_grad_batches=training_config.gradient_accumulation_steps,
        log_every_n_steps=training_config.log_every_n_steps,
        val_check_interval=training_config.val_check_interval,
        check_val_every_n_epoch=getattr(training_config, 'check_val_every_n_epoch', 1),
        deterministic=training_config.deterministic,
        benchmark=training_config.benchmark,
        sync_batchnorm=training_config.sync_batchnorm and devices > 1,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=2,
        profiler=profiler_instance,
        detect_anomaly=getattr(training_config, 'detect_anomaly', False)
    )
    
    return trainer


def train_vqvae(
    config,
    datamodule,
    max_epochs: int = 50
) -> str:
    """
    Pre-train VQ-VAE component.
    
    Args:
        config: ExperimentConfig object
        datamodule: Data module
        max_epochs: Maximum training epochs
        
    Returns:
        Path to best checkpoint
    """
    logger.info("=" * 60)
    logger.info("Starting VQ-VAE Pre-training")
    logger.info("=" * 60)
    
    # Create module
    module = VQVAELightning(config)
    
    # Checkpoint callback
    checkpoint_dir = Path(config.training.checkpoint_dir) / "vqvae"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='vqvae-{epoch:03d}-{val/recon_loss:.4f}',
        save_top_k=3,
        monitor='val/recon_loss',
        mode='min',
        save_last=True
    )
    
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(),
        RichProgressBar(),
        MetricLoggingCallback()
    ]
    
    # Create trainer
    n_gpus = config.training.n_gpus if torch.cuda.is_available() else 1
    
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=n_gpus,
        strategy=DDPStrategy(find_unused_parameters=False) if n_gpus > 1 else 'auto',
        precision='16-mixed' if config.training.use_amp else '32-true',
        callbacks=callbacks,
        logger=TensorBoardLogger(config.log_dir, name='vqvae'),
        gradient_clip_val=config.training.gradient_clip_val,
        log_every_n_steps=config.training.log_every_n_steps,
        deterministic=config.training.deterministic,
        sync_batchnorm=config.training.sync_batchnorm and n_gpus > 1
    )
    
    # Train
    with log_timing("VQ-VAE training"):
        trainer.fit(module, datamodule)
    
    best_path = checkpoint_callback.best_model_path
    logger.info(f"VQ-VAE training complete. Best checkpoint: {best_path}")
    
    return best_path


def train_skygpt_vitode(
    config,
    datamodule,
    vqvae_checkpoint: Optional[str] = None
) -> str:
    """
    Train full SkyGPT-ViTODE model.
    
    Args:
        config: ExperimentConfig object
        datamodule: Data module
        vqvae_checkpoint: Path to pre-trained VQ-VAE checkpoint
        
    Returns:
        Path to best checkpoint
    """
    logger.info("=" * 60)
    logger.info("Starting SkyGPT-ViTODE Training")
    logger.info("=" * 60)
    
    # Update config with VQ-VAE checkpoint
    if vqvae_checkpoint is not None:
        config.vqvae.pretrained_path = vqvae_checkpoint
        logger.info(f"Using pre-trained VQ-VAE: {vqvae_checkpoint}")
    
    # Create module
    module = SkyGPTViTODELightning(config)
    
    # Callbacks and loggers
    callbacks = create_callbacks(config)
    loggers = create_loggers(config)
    
    # Create trainer
    trainer = create_trainer(config, callbacks, loggers)
    
    # Resume from checkpoint if specified
    ckpt_path = getattr(config.training, 'resume_from_checkpoint', None)
    if ckpt_path and Path(ckpt_path).exists():
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    else:
        ckpt_path = None
    
    # Train
    with log_timing("SkyGPT-ViTODE training"):
        trainer.fit(module, datamodule, ckpt_path=ckpt_path)
    
    # Get best checkpoint path
    best_path = callbacks[0].best_model_path  # ModelCheckpoint is first
    logger.info(f"Training complete. Best checkpoint: {best_path}")
    
    return best_path


def test_model(
    config,
    datamodule,
    checkpoint_path: str
) -> Dict[str, float]:
    """
    Test trained model on test set.
    
    Args:
        config: ExperimentConfig object
        datamodule: Data module
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Dictionary with test metrics
    """
    logger.info("=" * 60)
    logger.info("Testing Model")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load model
    module = SkyGPTViTODELightning.load_from_checkpoint(
        checkpoint_path,
        config=config
    )
    
    # Create trainer for testing
    callbacks = [MetricLoggingCallback()]
    loggers = create_loggers(config)
    trainer = create_trainer(config, callbacks, loggers)
    
    # Test
    with log_timing("Model testing"):
        results = trainer.test(module, datamodule)
    
    return results[0] if results else {}


def run_ablation_study(
    base_config,
    datamodule,
    ablation_names: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Run ablation study experiments.
    
    Args:
        base_config: Base ExperimentConfig
        datamodule: Data module
        ablation_names: Specific ablations to run (None for all)
        
    Returns:
        Dictionary with results for each ablation
    """
    logger.info("=" * 60)
    logger.info("Starting Ablation Study")
    logger.info("=" * 60)
    
    # Import ablation configurations
    try:
        from config import get_ablation_configs
        all_ablations = get_ablation_configs()
    except ImportError:
        logger.error("Could not import ablation configurations")
        return {}
    
    # Filter if specific names provided
    if ablation_names is not None:
        ablations = {k: v for k, v in all_ablations.items() if k in ablation_names}
        missing = set(ablation_names) - set(ablations.keys())
        if missing:
            logger.warning(f"Unknown ablations: {missing}")
    else:
        ablations = all_ablations
    
    results = {}
    
    for name, ablation_config in ablations.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Running ablation: {name}")
        if hasattr(ablation_config, 'description'):
            logger.info(f"Description: {ablation_config.description}")
        logger.info('='*40)
        
        try:
            # Update experiment name
            ablation_config.experiment_name = f"{base_config.experiment_name}_{name}"
            
            # Train model
            checkpoint_path = train_skygpt_vitode(ablation_config, datamodule)
            
            # Test model
            test_results = test_model(ablation_config, datamodule, checkpoint_path)
            
            results[name] = {
                'checkpoint': checkpoint_path,
                'config_hash': ablation_config.get_config_hash() if hasattr(ablation_config, 'get_config_hash') else '',
                'status': 'success',
                'test_metrics': test_results
            }
            
            logger.info(f"Ablation {name} completed successfully")
            
        except Exception as e:
            logger.error(f"Ablation {name} failed: {e}")
            logger.error(traceback.format_exc())
            results[name] = {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        
        # Clear GPU memory between ablations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results summary
    results_dir = Path(base_config.results_dir) / 'ablations'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if YAML_AVAILABLE:
        results_path = results_dir / f'summary_{timestamp}.yaml'
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        results_path = results_dir / f'summary_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nAblation study complete. Results saved to: {results_path}")
    
    # Log summary
    logger.info("\nAblation Study Summary:")
    logger.info("-" * 40)
    for name, result in results.items():
        status = result['status']
        if status == 'success' and 'test_metrics' in result:
            metrics = result['test_metrics']
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float)])
            logger.info(f"  {name}: SUCCESS | {metric_str}")
        else:
            logger.info(f"  {name}: {status.upper()}")
    
    return results


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SkyGPT-ViTODE Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--experiment-name', type=str, default='skygpt_vitode',
        help='Experiment name'
    )
    
    # Training mode
    parser.add_argument(
        '--mode', type=str, default='full',
        choices=['vqvae', 'full', 'ablation', 'test'],
        help='Training mode'
    )
    
    # Data
    parser.add_argument(
        '--data-path', type=str, default=None,
        help="Path to SKIPP'D dataset"
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--num-workers', type=int, default=None,
        help='Number of data loading workers'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--max-epochs', type=int, default=None,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--n-gpus', type=int, default=None,
        help='Number of GPUs'
    )
    parser.add_argument(
        '--gradient-accumulation', type=int, default=None,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=None,
        help='Weight decay'
    )
    
    # Checkpoints
    parser.add_argument(
        '--vqvae-checkpoint', type=str, default=None,
        help='Path to pre-trained VQ-VAE checkpoint'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    
    # Ablation study
    parser.add_argument(
        '--ablations', type=str, nargs='+', default=None,
        help='Specific ablation configurations to run'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed'
    )
    
    # Debugging
    parser.add_argument(
        '--fast-dev-run', action='store_true',
        help='Run quick debug iteration'
    )
    parser.add_argument(
        '--overfit-batches', type=float, default=0.0,
        help='Overfit on fraction of data for debugging'
    )
    parser.add_argument(
        '--profiler', type=str, default=None,
        choices=['simple', 'advanced'],
        help='Profiler type'
    )
    parser.add_argument(
        '--detect-anomaly', action='store_true',
        help='Enable anomaly detection for debugging'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Check Lightning availability
    if not LIGHTNING_AVAILABLE:
        print("ERROR: PyTorch Lightning is required.")
        print("Install with: pip install pytorch-lightning")
        sys.exit(1)
    
    # Import configuration
    try:
        from config import ExperimentConfig, setup_logging, set_seed
    except ImportError as e:
        print(f"ERROR: Could not import config module: {e}")
        sys.exit(1)
    
    # Load or create configuration
    if args.config is not None:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig(experiment_name=args.experiment_name)
    
    # Override with command line arguments
    if args.data_path is not None:
        config.data.data_path = args.data_path
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.n_gpus is not None:
        config.training.n_gpus = args.n_gpus
    if args.gradient_accumulation is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.seed is not None:
        config.training.seed = args.seed
    if args.resume is not None:
        config.training.resume_from_checkpoint = args.resume
    if args.detect_anomaly:
        config.training.detect_anomaly = True
    
    # Set random seed
    set_seed(config.training.seed, config.training.deterministic)
    seed_everything(config.training.seed, workers=True)
    
    # Setup logging
    setup_logging(config.log_dir, config.experiment_name)
    setup_training_logging(config.log_dir, config.experiment_name, level=args.log_level)
    
    # Log configuration
    effective_batch = (
        config.data.batch_size *
        config.training.gradient_accumulation_steps *
        max(1, config.training.n_gpus)
    )
    
    logger.info("=" * 60)
    logger.info("SkyGPT-ViTODE Training")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Config hash: {config.get_config_hash() if hasattr(config, 'get_config_hash') else 'N/A'}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Lightning: {pl.__version__ if pl else 'N/A'}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    logger.info(f"GPUs to use: {config.training.n_gpus}")
    logger.info(f"Batch size: {config.data.batch_size} per GPU")
    logger.info(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Max epochs: {config.training.max_epochs}")
    logger.info(f"Scheduler: {config.training.scheduler}")
    logger.info(f"Optimizer: {config.training.optimizer}")
    
    # Save configuration
    try:
        config_path = config.save()
        logger.info(f"Configuration saved to: {config_path}")
    except Exception as e:
        logger.warning(f"Could not save configuration: {e}")
    
    # Create data module
    try:
        from data import create_datamodule
        datamodule = create_datamodule(config)
    except ImportError as e:
        logger.error(f"Could not import data module: {e}")
        sys.exit(1)
    
    try:
        datamodule.prepare_data()
        datamodule.setup()
        
        logger.info(f"Train samples: {datamodule.n_train_samples}")
        logger.info(f"Val samples: {datamodule.n_val_samples}")
        logger.info(f"Test samples: {datamodule.n_test_samples}")
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.info("Please ensure the SKIPP'D dataset is available at the configured path.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error setting up data: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Run training based on mode
    try:
        if args.mode == 'vqvae':
            max_epochs = args.max_epochs or 50
            train_vqvae(config, datamodule, max_epochs=max_epochs)
            
        elif args.mode == 'full':
            # Pre-train VQ-VAE if no checkpoint provided
            vqvae_checkpoint = args.vqvae_checkpoint or getattr(config.vqvae, 'pretrained_path', None)
            
            if vqvae_checkpoint is None or not Path(vqvae_checkpoint).exists():
                logger.info("No VQ-VAE checkpoint provided. Pre-training VQ-VAE first...")
                vqvae_checkpoint = train_vqvae(config, datamodule, max_epochs=50)
            
            # Train full model
            train_skygpt_vitode(config, datamodule, vqvae_checkpoint)
            
        elif args.mode == 'ablation':
            run_ablation_study(config, datamodule, args.ablations)
            
        elif args.mode == 'test':
            # Load and test model
            if args.resume is None:
                logger.error("Test mode requires --resume checkpoint path")
                sys.exit(1)
            
            test_model(config, datamodule, args.resume)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()