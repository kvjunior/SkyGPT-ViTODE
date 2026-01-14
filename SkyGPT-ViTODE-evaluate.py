"""
SkyGPT-ViTODE Evaluation and Visualization Module
==================================================

This module provides comprehensive evaluation capabilities for the SkyGPT-ViTODE
framework for probabilistic ultra-short-term solar forecasting.

Features:
    - Probabilistic forecasting metrics (CRPS, Winkler Score, Reliability)
    - Multiple calibration assessment methods (reliability diagram, PIT histogram)
    - Conformal prediction calibration with multiple coverage levels
    - Statistical significance testing (Diebold-Mariano, paired t-test, Wilcoxon)
    - Bootstrap confidence intervals for all metrics
    - Publication-quality figure generation (Times New Roman, 300 DPI)
    - Multi-horizon and temporal aggregation analysis
    - Skill score computation against persistence and climatology baselines
    - Comprehensive result export (LaTeX, CSV, JSON)

"""

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
from contextlib import contextmanager
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from scipy import stats
    from scipy.special import ndtri, erf
    from scipy.integrate import quad
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical tests will be disabled.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x

# Plotting imports with error handling
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization will be disabled.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# ============================================================================
# Constants
# ============================================================================

EPS = 1e-8
LOG_EPS = 1e-10
SQRT_2 = np.sqrt(2)
SQRT_PI = np.sqrt(np.pi)
SQRT_2PI = np.sqrt(2 * np.pi)

# Gaussian quantiles for common coverage levels
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


# ============================================================================
# Logging and Configuration
# ============================================================================

logger = logging.getLogger('SkyGPT-ViTODE.Evaluate')


def setup_evaluation_logging(
    log_dir: str,
    experiment_name: str,
    level: str = "INFO"
) -> logging.Logger:
    """Setup evaluation-specific logging."""
    log_path = Path(log_dir) / experiment_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"evaluate_{timestamp}.log"
    
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    
    return logger


# Publication-quality figure settings
def setup_matplotlib_style(use_latex: bool = False):
    """Configure matplotlib for publication-quality figures."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    style_dict = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.constrained_layout.use': True,
        'mathtext.fontset': 'cm',
    }
    
    if use_latex:
        style_dict.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        })
    
    plt.rcParams.update(style_dict)


# Color palettes
class ColorPalette:
    """Professional color palettes for visualization."""
    
    # Default palette (colorblind-friendly)
    DEFAULT = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'tertiary': '#F18F01',
        'quaternary': '#C73E1D',
        'quinary': '#3A7D44',
        'light': '#E8E8E8',
        'dark': '#1A1A2E',
        'success': '#28A745',
        'warning': '#FFC107',
        'error': '#DC3545',
    }
    
    # IEEE/ACM style palette
    ACADEMIC = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'tertiary': '#2ca02c',
        'quaternary': '#d62728',
        'quinary': '#9467bd',
        'senary': '#8c564b',
        'septenary': '#e377c2',
        'octonary': '#7f7f7f',
    }
    
    # Viridis-based for heatmaps
    SEQUENTIAL = 'viridis'
    DIVERGING = 'RdBu_r'
    
    @classmethod
    def get_colors(cls, n: int, palette: str = 'default') -> List[str]:
        """Get n colors from specified palette."""
        if palette == 'default':
            colors = list(cls.DEFAULT.values())
        elif palette == 'academic':
            colors = list(cls.ACADEMIC.values())
        else:
            colors = list(cls.DEFAULT.values())
        
        # Cycle if more colors needed
        return [colors[i % len(colors)] for i in range(n)]


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class MetricResult:
    """Container for a single metric with confidence interval."""
    value: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None
    std_error: Optional[float] = None
    n_samples: int = 0
    
    def __str__(self) -> str:
        if self.lower_ci is not None and self.upper_ci is not None:
            return f"{self.value:.4f} [{self.lower_ci:.4f}, {self.upper_ci:.4f}]"
        return f"{self.value:.4f}"
    
    def to_latex(self, precision: int = 3, bold: bool = False) -> str:
        """Format for LaTeX table."""
        fmt = f".{precision}f"
        val_str = f"{self.value:{fmt}}"
        if bold:
            val_str = f"\\textbf{{{val_str}}}"
        if self.lower_ci is not None:
            ci_str = f"$\\pm${abs(self.value - self.lower_ci):{fmt}}"
            return f"{val_str} {ci_str}"
        return val_str


@dataclass
class EvaluationResults:
    """Comprehensive container for evaluation results."""
    
    # Point prediction metrics
    mae: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    rmse: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    mbe: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    r2: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    
    # Normalized metrics
    nmae: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    nrmse: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    
    # Probabilistic metrics
    crps: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    crps_skill_persistence: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    crps_skill_climatology: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    
    # Winkler scores at different levels
    winkler_80: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    winkler_90: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    winkler_95: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    
    # Calibration metrics
    coverage_80: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    coverage_90: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    coverage_95: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    calibration_error: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    
    # Sharpness metrics
    avg_interval_width_90: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    pinaw: MetricResult = field(default_factory=lambda: MetricResult(0.0))
    
    # Statistical significance
    dm_vs_persistence: Optional[Tuple[float, float]] = None
    dm_vs_climatology: Optional[Tuple[float, float]] = None
    
    # Metadata
    n_samples: int = 0
    model_name: str = ""
    evaluation_timestamp: str = ""
    config_hash: str = ""
    
    def __post_init__(self):
        if not self.evaluation_timestamp:
            self.evaluation_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with nested structure for MetricResults."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict) and 'value' in value:
                result[key] = value
            else:
                result[key] = value
        return result
    
    def to_flat_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary with only values."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict) and 'value' in value:
                result[key] = value['value']
            elif isinstance(value, (int, float)):
                result[key] = value
        return result
    
    def to_latex_row(
        self,
        precision: int = 3,
        metrics: List[str] = None,
        best_values: Dict[str, float] = None
    ) -> str:
        """Generate LaTeX table row with optional best-value highlighting."""
        if metrics is None:
            metrics = ['mae', 'rmse', 'crps', 'winkler_90', 'coverage_90']
        
        values = []
        for metric in metrics:
            attr = getattr(self, metric, None)
            if attr is None:
                values.append("--")
                continue
            
            if isinstance(attr, MetricResult):
                is_best = (
                    best_values is not None and
                    metric in best_values and
                    abs(attr.value - best_values[metric]) < 1e-6
                )
                values.append(attr.to_latex(precision, bold=is_best))
            else:
                values.append(f"{attr:.{precision}f}")
        
        return f"{self.model_name} & " + " & ".join(values) + " \\\\"


@dataclass
class ReliabilityData:
    """Container for reliability diagram data."""
    expected_coverage: np.ndarray
    empirical_coverage: np.ndarray
    n_samples: int
    calibration_error: float
    pit_values: Optional[np.ndarray] = None


@dataclass
class ComparisonResult:
    """Container for model comparison results."""
    model1_name: str
    model2_name: str
    dm_statistic: float
    dm_pvalue: float
    ttest_statistic: float
    ttest_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    mean_diff: float
    mean_diff_ci: Tuple[float, float]
    significant_dm: bool
    significant_ttest: bool
    significant_wilcoxon: bool


# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_crps_gaussian(
    mean: np.ndarray,
    std: np.ndarray,
    observation: np.ndarray
) -> np.ndarray:
    """
    Compute closed-form CRPS for Gaussian distribution.
    
    CRPS(N(μ,σ²), y) = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    
    where z = (y - μ)/σ, Φ is standard normal CDF, φ is standard normal PDF.
    
    Reference:
        Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
        prediction, and estimation. JASA, 102(477), 359-378.
    
    Args:
        mean: Predicted means (N,)
        std: Predicted standard deviations (N,)
        observation: Observed values (N,)
        
    Returns:
        CRPS values (N,)
    """
    std_safe = np.maximum(std, EPS)
    z = (observation - mean) / std_safe
    
    # Standard normal PDF and CDF
    phi = np.exp(-0.5 * z ** 2) / SQRT_2PI
    Phi = 0.5 * (1 + erf(z / SQRT_2))
    
    # CRPS formula
    crps = std_safe * (z * (2 * Phi - 1) + 2 * phi - 1 / SQRT_PI)
    
    return crps


def compute_crps_ensemble(
    samples: np.ndarray,
    observation: np.ndarray
) -> np.ndarray:
    """
    Compute empirical CRPS from ensemble samples.
    
    CRPS = E|X - y| - 0.5 * E|X - X'|
    
    Uses the efficient formula via order statistics:
    E|X - X'| = (2/M²) * Σᵢ (2i - M - 1) * X₍ᵢ₎
    
    Args:
        samples: Ensemble samples (N, M) where M is ensemble size
        observation: Observed values (N,)
        
    Returns:
        CRPS values (N,)
    """
    n_samples, m = samples.shape
    
    # Term 1: E|X - y|
    obs_expanded = observation[:, np.newaxis]
    term1 = np.mean(np.abs(samples - obs_expanded), axis=1)
    
    # Term 2: E|X - X'| via order statistics
    samples_sorted = np.sort(samples, axis=1)
    
    # Weights: (2i - M - 1) / M² for i = 1, ..., M
    i = np.arange(1, m + 1)
    weights = (2 * i - m - 1) / (m * m)
    
    term2 = np.sum(weights * samples_sorted, axis=1)
    
    # CRPS = E|X - y| - 0.5 * E|X - X'|
    # Note: term2 already computes the full expectation
    crps = term1 - np.abs(term2)
    
    return crps


def compute_crps(
    predictions: np.ndarray,
    observations: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    samples: Optional[np.ndarray] = None
) -> float:
    """
    Compute mean CRPS using best available method.
    
    Args:
        predictions: Point predictions (N,)
        observations: Ground truth values (N,)
        uncertainties: Standard deviations (N,) for Gaussian
        samples: Ensemble samples (N, M) for empirical
        
    Returns:
        Mean CRPS value
    """
    if samples is not None:
        crps = compute_crps_ensemble(samples, observations)
    elif uncertainties is not None:
        crps = compute_crps_gaussian(predictions, uncertainties, observations)
    else:
        # Deterministic: CRPS = MAE
        crps = np.abs(predictions - observations)
    
    return float(np.mean(crps))


def compute_winkler_score(
    lower: np.ndarray,
    upper: np.ndarray,
    observations: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Compute Winkler Score for prediction intervals.
    
    W(L, U, y) = (U - L) + (2/α) * (L - y) * I(y < L) + (2/α) * (y - U) * I(y > U)
    
    Args:
        lower: Lower bounds (N,)
        upper: Upper bounds (N,)
        observations: Ground truth (N,)
        alpha: Significance level (1 - coverage)
        
    Returns:
        Mean Winkler Score
    """
    delta = upper - lower
    
    below_mask = observations < lower
    above_mask = observations > upper
    
    below_penalty = (2 / alpha) * (lower - observations) * below_mask
    above_penalty = (2 / alpha) * (observations - upper) * above_mask
    
    score = delta + below_penalty + above_penalty
    return float(np.mean(score))


def compute_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    observations: np.ndarray
) -> float:
    """
    Compute empirical coverage probability.
    
    Args:
        lower: Lower bounds (N,)
        upper: Upper bounds (N,)
        observations: Ground truth (N,)
        
    Returns:
        Coverage probability in [0, 1]
    """
    covered = (observations >= lower) & (observations <= upper)
    return float(np.mean(covered))


def compute_pinaw(
    lower: np.ndarray,
    upper: np.ndarray,
    y_range: float
) -> float:
    """
    Compute Prediction Interval Normalized Average Width.
    
    PINAW = mean(U - L) / (y_max - y_min)
    
    Args:
        lower: Lower bounds (N,)
        upper: Upper bounds (N,)
        y_range: Range of target variable
        
    Returns:
        PINAW value
    """
    widths = upper - lower
    return float(np.mean(widths) / (y_range + EPS))


def compute_pit_values(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    observations: np.ndarray
) -> np.ndarray:
    """
    Compute Probability Integral Transform (PIT) values.
    
    PIT(y) = F(y | μ, σ) = Φ((y - μ) / σ)
    
    For well-calibrated forecasts, PIT values should be uniformly distributed.
    
    Args:
        predictions: Point predictions (N,)
        uncertainties: Standard deviations (N,)
        observations: Ground truth (N,)
        
    Returns:
        PIT values in [0, 1]
    """
    z = (observations - predictions) / (uncertainties + EPS)
    pit = 0.5 * (1 + erf(z / SQRT_2))
    return pit


def compute_reliability_diagram(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    observations: np.ndarray,
    n_bins: int = 10
) -> ReliabilityData:
    """
    Compute reliability diagram data.
    
    Args:
        predictions: Point predictions (N,)
        uncertainties: Standard deviations (N,)
        observations: Ground truth (N,)
        n_bins: Number of probability bins
        
    Returns:
        ReliabilityData object
    """
    # Expected coverage levels
    expected_levels = np.linspace(0.1, 0.95, n_bins)
    
    # Compute empirical coverage at each level
    empirical_coverage = []
    for level in expected_levels:
        quantile = GAUSSIAN_QUANTILES.get(
            level,
            ndtri((1 + level) / 2) if SCIPY_AVAILABLE else 1.645
        )
        lower = predictions - quantile * uncertainties
        upper = predictions + quantile * uncertainties
        coverage = compute_coverage(lower, upper, observations)
        empirical_coverage.append(coverage)
    
    empirical_coverage = np.array(empirical_coverage)
    
    # Calibration error (mean absolute deviation from diagonal)
    cal_error = float(np.mean(np.abs(expected_levels - empirical_coverage)))
    
    # PIT values
    pit = compute_pit_values(predictions, uncertainties, observations)
    
    return ReliabilityData(
        expected_coverage=expected_levels,
        empirical_coverage=empirical_coverage,
        n_samples=len(predictions),
        calibration_error=cal_error,
        pit_values=pit
    )


def compute_skill_score(
    model_errors: np.ndarray,
    reference_errors: np.ndarray
) -> float:
    """
    Compute skill score relative to reference.
    
    SS = 1 - MSE_model / MSE_reference
    
    Args:
        model_errors: Model forecast errors
        reference_errors: Reference forecast errors
        
    Returns:
        Skill score (positive = better than reference)
    """
    mse_model = np.mean(model_errors ** 2)
    mse_ref = np.mean(reference_errors ** 2)
    
    if mse_ref < EPS:
        return 0.0
    
    return float(1 - mse_model / mse_ref)


# ============================================================================
# Statistical Testing Functions
# ============================================================================

def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    loss: str = 'squared'
) -> Tuple[float, float]:
    """
    Perform Diebold-Mariano test for comparing forecast accuracy.
    
    Tests H₀: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)
    
    Uses Newey-West HAC estimator for variance with h-1 lags.
    
    Reference:
        Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
        Journal of Business & Economic Statistics, 13(3), 253-263.
    
    Args:
        errors1: Forecast errors from model 1 (N,)
        errors2: Forecast errors from model 2 (N,)
        h: Forecast horizon
        loss: Loss function ('squared' or 'absolute')
        
    Returns:
        Tuple of (DM statistic, two-sided p-value)
    """
    # Compute loss differentials
    if loss == 'squared':
        d = errors1 ** 2 - errors2 ** 2
    else:
        d = np.abs(errors1) - np.abs(errors2)
    
    n = len(d)
    d_bar = np.mean(d)
    
    # Newey-West HAC variance estimator
    # γ₀ = Var(d)
    gamma_0 = np.var(d, ddof=1)
    
    # Autocovariances with Bartlett weights
    gamma_sum = 0.0
    for k in range(1, h):
        weight = 1 - k / h  # Bartlett kernel
        d_centered = d - d_bar
        gamma_k = np.mean(d_centered[k:] * d_centered[:-k])
        gamma_sum += 2 * weight * gamma_k
    
    # Long-run variance
    long_run_var = gamma_0 + gamma_sum
    
    # Variance of d_bar
    var_d_bar = long_run_var / n
    
    # DM statistic
    if var_d_bar <= 0:
        return 0.0, 1.0
    
    dm_stat = d_bar / np.sqrt(var_d_bar)
    
    # Two-sided p-value (asymptotically N(0,1))
    if SCIPY_AVAILABLE:
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    else:
        # Fallback approximation
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(dm_stat) / SQRT_2)))
    
    return float(dm_stat), float(p_value)


def paired_ttest(
    errors1: np.ndarray,
    errors2: np.ndarray
) -> Tuple[float, float]:
    """
    Perform paired t-test for comparing forecast errors.
    
    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        
    Returns:
        Tuple of (t-statistic, two-sided p-value)
    """
    if not SCIPY_AVAILABLE:
        return 0.0, 1.0
    
    diff = np.abs(errors1) - np.abs(errors2)
    t_stat, p_value = stats.ttest_1samp(diff, 0)
    
    return float(t_stat), float(p_value)


def wilcoxon_test(
    errors1: np.ndarray,
    errors2: np.ndarray
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric).
    
    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        
    Returns:
        Tuple of (test statistic, two-sided p-value)
    """
    if not SCIPY_AVAILABLE:
        return 0.0, 1.0
    
    diff = np.abs(errors1) - np.abs(errors2)
    
    # Remove zeros
    diff = diff[diff != 0]
    
    if len(diff) < 10:
        return 0.0, 1.0
    
    try:
        stat, p_value = stats.wilcoxon(diff)
        return float(stat), float(p_value)
    except ValueError:
        return 0.0, 1.0


def bootstrap_metric(
    data: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> MetricResult:
    """
    Compute metric with bootstrap confidence interval.
    
    Args:
        data: Input data array (or tuple of arrays)
        metric_fn: Function to compute metric
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed
        
    Returns:
        MetricResult with value and confidence interval
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Handle single array or multiple arrays
    if isinstance(data, tuple):
        n = len(data[0])
    else:
        n = len(data)
        data = (data,)
    
    # Point estimate
    point_estimate = metric_fn(*data)
    
    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        sample = tuple(arr[indices] for arr in data)
        try:
            stat = metric_fn(*sample)
            bootstrap_stats.append(stat)
        except (ValueError, ZeroDivisionError):
            continue
    
    if len(bootstrap_stats) < 100:
        return MetricResult(value=point_estimate, n_samples=n)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile confidence interval
    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    std_error = float(np.std(bootstrap_stats))
    
    return MetricResult(
        value=float(point_estimate),
        lower_ci=lower,
        upper_ci=upper,
        std_error=std_error,
        n_samples=n
    )


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        List of (adjusted_p, is_significant) tuples
    """
    m = len(p_values)
    adjusted_alpha = alpha / m
    
    return [
        (min(p * m, 1.0), p < adjusted_alpha)
        for p in p_values
    ]


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[float, bool]]:
    """
    Apply Holm-Bonferroni correction (step-down procedure).
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        List of (adjusted_p, is_significant) tuples
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    
    results = [None] * m
    
    for rank, idx in enumerate(sorted_indices):
        adjusted_p = min(p_values[idx] * (m - rank), 1.0)
        threshold = alpha / (m - rank)
        significant = p_values[idx] < threshold
        results[idx] = (adjusted_p, significant)
    
    return results


# ============================================================================
# Baseline Models for Comparison
# ============================================================================

class PersistenceBaseline:
    """
    Persistence forecast baseline.
    
    Predicts that future value equals the most recent observation.
    """
    
    @staticmethod
    def predict(history: np.ndarray) -> np.ndarray:
        """Return last observed value."""
        return history[:, -1]
    
    @staticmethod
    def compute_errors(history: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute persistence forecast errors."""
        return history[:, -1] - target


class ClimatologyBaseline:
    """
    Climatology forecast baseline.
    
    Predicts historical mean as the forecast.
    """
    
    def __init__(self):
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.n: int = 0
    
    def fit(self, values: np.ndarray):
        """Fit climatology from training data."""
        self.mean = float(np.mean(values))
        self.std = float(np.std(values))
        self.n = len(values)
    
    def predict(self, n_samples: int) -> np.ndarray:
        """Return climatology prediction."""
        if self.mean is None:
            raise ValueError("Climatology not fitted")
        return np.full(n_samples, self.mean)
    
    def compute_errors(self, target: np.ndarray) -> np.ndarray:
        """Compute climatology forecast errors."""
        if self.mean is None:
            raise ValueError("Climatology not fitted")
        return self.mean - target


class SmartPersistence:
    """
    Smart persistence using clear-sky index.
    
    Predicts: P(t+h) = P(t) * CSI(t+h) / CSI(t)
    """
    
    @staticmethod
    def predict(
        history: np.ndarray,
        clear_sky_now: np.ndarray,
        clear_sky_future: np.ndarray
    ) -> np.ndarray:
        """Compute smart persistence forecast."""
        csi_ratio = clear_sky_future / (clear_sky_now + EPS)
        return history[:, -1] * csi_ratio


# ============================================================================
# Model Evaluator Class
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Performs full evaluation pipeline including:
    - Batch inference with progress tracking
    - Point and probabilistic metric computation
    - Conformal calibration
    - Statistical significance testing
    - Bootstrap confidence intervals
    - Comparison with baselines
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            config: Experiment configuration
            device: Computation device
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Results storage
        self.predictions: Optional[np.ndarray] = None
        self.uncertainties: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.samples: Optional[np.ndarray] = None
        self.pv_history: Optional[np.ndarray] = None
        self.timestamps: Optional[np.ndarray] = None
        
        # Baselines
        self.persistence = PersistenceBaseline()
        self.climatology = ClimatologyBaseline()
        
        # Denormalization parameters
        self.pv_max = getattr(config.data, 'pv_max', 1.0)
        self.pv_min = getattr(config.data, 'pv_min', 0.0)
        self.pv_range = self.pv_max - self.pv_min
    
    def _denormalize(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized values to physical units."""
        return values * self.pv_range + self.pv_min
    
    @torch.no_grad()
    def run_inference(
        self,
        dataloader: DataLoader,
        n_samples: int = 50,
        show_progress: bool = True
    ) -> None:
        """
        Run inference on dataset.
        
        Args:
            dataloader: Test data loader
            n_samples: Number of samples for probabilistic prediction
            show_progress: Show progress bar
        """
        logger.info("Running inference...")
        
        predictions_list = []
        uncertainties_list = []
        targets_list = []
        samples_list = []
        history_list = []
        
        iterator = tqdm(dataloader, desc="Inference") if show_progress and TQDM_AVAILABLE else dataloader
        
        for batch in iterator:
            video = batch['video'].to(self.device)
            pv_target = batch['pv_target'].to(self.device)
            pv_history = batch.get('pv_history')
            
            # Forward pass
            output = self.model(video, pv_target)
            pv_output = output['pv_output']
            
            # Extract predictions based on output type
            pred_mean, pred_std, batch_samples = self._extract_predictions(
                pv_output, n_samples
            )
            
            # Store results (still normalized)
            predictions_list.append(pred_mean.cpu().numpy())
            uncertainties_list.append(pred_std.cpu().numpy())
            targets_list.append(pv_target.cpu().numpy())
            samples_list.append(batch_samples.cpu().numpy())
            
            if pv_history is not None:
                history_list.append(pv_history.numpy())
        
        # Concatenate and denormalize
        self.predictions = self._denormalize(np.concatenate(predictions_list))
        self.uncertainties = np.concatenate(uncertainties_list) * self.pv_range
        self.targets = self._denormalize(np.concatenate(targets_list))
        self.samples = self._denormalize(np.concatenate(samples_list))
        
        if history_list:
            self.pv_history = self._denormalize(np.concatenate(history_list))
        
        logger.info(f"Inference complete. {len(self.predictions)} samples processed.")
    
    def _extract_predictions(
        self,
        pv_output: Dict[str, Tensor],
        n_samples: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract mean, std, and samples from model output."""
        output_type = getattr(self.config.vit, 'output_type', 'gaussian')
        
        if output_type == "mdn":
            weights = pv_output['weights']
            means = pv_output['means']
            scales = pv_output['scales']
            
            # Point prediction (weighted mean)
            pred_mean = (weights * means).sum(dim=-1)
            
            # Uncertainty (mixture std via law of total variance)
            var_within = (weights * scales ** 2).sum(dim=-1)
            var_between = (weights * (means - pred_mean.unsqueeze(-1)) ** 2).sum(dim=-1)
            pred_std = torch.sqrt(var_within + var_between + EPS)
            
            # Sample from mixture (vectorized)
            batch_samples = self._sample_mdn_vectorized(
                weights, means, scales, n_samples
            )
            
        elif output_type == "gaussian":
            pred_mean = pv_output['mean']
            pred_std = pv_output['std']
            batch_samples = pred_mean.unsqueeze(1) + pred_std.unsqueeze(1) * \
                           torch.randn(pred_mean.shape[0], n_samples, device=self.device)
            
        else:  # deterministic
            pred_mean = pv_output.get('prediction', pv_output.get('mean'))
            pred_std = torch.zeros_like(pred_mean)
            batch_samples = pred_mean.unsqueeze(1).expand(-1, n_samples)
        
        return pred_mean, pred_std, batch_samples
    
    def _sample_mdn_vectorized(
        self,
        weights: Tensor,
        means: Tensor,
        scales: Tensor,
        n_samples: int
    ) -> Tensor:
        """Vectorized sampling from MDN."""
        batch_size, n_components = weights.shape
        
        # Sample component indices
        component_indices = torch.multinomial(
            weights, n_samples, replacement=True
        )  # (B, n_samples)
        
        # Gather means and scales
        # Expand indices for gather
        indices_expanded = component_indices.unsqueeze(-1)
        
        selected_means = torch.gather(
            means.unsqueeze(1).expand(-1, n_samples, -1),
            dim=2,
            index=indices_expanded
        ).squeeze(-1)  # (B, n_samples)
        
        selected_scales = torch.gather(
            scales.unsqueeze(1).expand(-1, n_samples, -1),
            dim=2,
            index=indices_expanded
        ).squeeze(-1)  # (B, n_samples)
        
        # Sample
        samples = selected_means + selected_scales * torch.randn_like(selected_means)
        
        return samples
    
    def calibrate_conformal(
        self,
        calibration_ratio: float = 0.2,
        coverage_levels: List[float] = None
    ) -> Dict[float, float]:
        """
        Calibrate conformal predictor for multiple coverage levels.
        
        Args:
            calibration_ratio: Ratio of data for calibration
            coverage_levels: Target coverage levels
            
        Returns:
            Dictionary mapping coverage level to calibrated quantile
        """
        if coverage_levels is None:
            coverage_levels = getattr(
                self.config.conformal, 'coverage_levels', [0.90]
            )
        
        n_total = len(self.predictions)
        n_cal = int(n_total * calibration_ratio)
        
        # Use first portion for calibration
        cal_pred = self.predictions[:n_cal]
        cal_unc = self.uncertainties[:n_cal]
        cal_target = self.targets[:n_cal]
        
        # Compute nonconformity scores
        scores = np.abs(cal_target - cal_pred) / (cal_unc + EPS)
        
        # Calibrate for each coverage level
        calibrated_quantiles = {}
        for level in coverage_levels:
            # Finite sample correction
            q_level = np.ceil((n_cal + 1) * level) / n_cal
            q_level = min(q_level, 1.0)
            
            quantile = float(np.quantile(scores, q_level))
            calibrated_quantiles[level] = quantile
            
            logger.info(
                f"Conformal calibration: {100*level:.0f}% coverage → "
                f"quantile = {quantile:.4f}"
            )
        
        return calibrated_quantiles
    
    def compute_metrics(
        self,
        calibrated_quantiles: Dict[float, float] = None,
        model_name: str = "",
        compute_bootstrap: bool = True,
        n_bootstrap: int = 1000
    ) -> EvaluationResults:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            calibrated_quantiles: Calibrated conformal quantiles
            model_name: Name for results
            compute_bootstrap: Compute bootstrap CIs
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            EvaluationResults object
        """
        pred = self.predictions
        unc = self.uncertainties
        target = self.targets
        samples = self.samples
        
        results = EvaluationResults(
            model_name=model_name or self.config.experiment_name,
            n_samples=len(pred)
        )
        
        # Point prediction metrics
        errors = pred - target
        
        # MAE
        if compute_bootstrap:
            results.mae = bootstrap_metric(
                (pred, target),
                lambda p, t: np.mean(np.abs(p - t)),
                n_bootstrap
            )
        else:
            results.mae = MetricResult(float(np.mean(np.abs(errors))), n_samples=len(pred))
        
        # RMSE
        if compute_bootstrap:
            results.rmse = bootstrap_metric(
                (pred, target),
                lambda p, t: np.sqrt(np.mean((p - t) ** 2)),
                n_bootstrap
            )
        else:
            results.rmse = MetricResult(float(np.sqrt(np.mean(errors ** 2))), n_samples=len(pred))
        
        # MBE (Mean Bias Error)
        results.mbe = MetricResult(float(np.mean(errors)), n_samples=len(pred))
        
        # R² score
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - ss_res / (ss_tot + EPS)
        results.r2 = MetricResult(float(r2), n_samples=len(pred))
        
        # Normalized metrics
        results.nmae = MetricResult(
            float(np.mean(np.abs(errors)) / (np.mean(target) + EPS)),
            n_samples=len(pred)
        )
        results.nrmse = MetricResult(
            float(np.sqrt(np.mean(errors ** 2)) / (np.mean(target) + EPS)),
            n_samples=len(pred)
        )
        
        # CRPS
        if compute_bootstrap:
            results.crps = bootstrap_metric(
                (pred, target, unc, samples),
                lambda p, t, u, s: compute_crps(p, t, u, s),
                n_bootstrap
            )
        else:
            crps_val = compute_crps(pred, target, unc, samples)
            results.crps = MetricResult(crps_val, n_samples=len(pred))
        
        # CRPS skill scores
        if self.pv_history is not None:
            persistence_errors = self.persistence.compute_errors(self.pv_history, target)
            persistence_crps = np.mean(np.abs(persistence_errors))
            skill = 1 - results.crps.value / (persistence_crps + EPS)
            results.crps_skill_persistence = MetricResult(skill, n_samples=len(pred))
            
            # Climatology skill
            self.climatology.fit(target)
            clim_errors = self.climatology.compute_errors(target)
            clim_crps = np.mean(np.abs(clim_errors))
            skill_clim = 1 - results.crps.value / (clim_crps + EPS)
            results.crps_skill_climatology = MetricResult(skill_clim, n_samples=len(pred))
        
        # Prediction intervals and Winkler scores
        for level, name_suffix in [(0.80, '80'), (0.90, '90'), (0.95, '95')]:
            if calibrated_quantiles and level in calibrated_quantiles:
                q = calibrated_quantiles[level]
            else:
                q = GAUSSIAN_QUANTILES.get(level, 1.645)
            
            lower = pred - q * unc
            upper = pred + q * unc
            
            # Winkler score
            alpha = 1 - level
            winkler = compute_winkler_score(lower, upper, target, alpha)
            setattr(results, f'winkler_{name_suffix}', 
                   MetricResult(winkler, n_samples=len(pred)))
            
            # Coverage
            coverage = compute_coverage(lower, upper, target)
            setattr(results, f'coverage_{name_suffix}',
                   MetricResult(coverage, n_samples=len(pred)))
        
        # Average interval width (90%)
        q90 = calibrated_quantiles.get(0.90, 1.645) if calibrated_quantiles else 1.645
        lower_90 = pred - q90 * unc
        upper_90 = pred + q90 * unc
        results.avg_interval_width_90 = MetricResult(
            float(np.mean(upper_90 - lower_90)),
            n_samples=len(pred)
        )
        
        # PINAW
        y_range = float(np.max(target) - np.min(target))
        results.pinaw = MetricResult(
            compute_pinaw(lower_90, upper_90, y_range),
            n_samples=len(pred)
        )
        
        # Calibration error
        rel_data = compute_reliability_diagram(pred, unc, target)
        results.calibration_error = MetricResult(
            rel_data.calibration_error,
            n_samples=len(pred)
        )
        
        # Statistical tests vs baselines
        if self.pv_history is not None:
            persistence_errors = self.persistence.compute_errors(self.pv_history, target)
            dm_stat, dm_p = diebold_mariano_test(errors, persistence_errors)
            results.dm_vs_persistence = (dm_stat, dm_p)
        
        return results
    
    def compare_models(
        self,
        other_predictions: np.ndarray,
        other_name: str = "Other",
        significance_level: float = 0.05
    ) -> ComparisonResult:
        """
        Compare this model with another using multiple statistical tests.
        
        Args:
            other_predictions: Predictions from other model
            other_name: Name of other model
            significance_level: Significance level for tests
            
        Returns:
            ComparisonResult object
        """
        errors1 = self.predictions - self.targets
        errors2 = other_predictions - self.targets
        
        # Diebold-Mariano test
        dm_stat, dm_p = diebold_mariano_test(errors1, errors2)
        
        # Paired t-test
        t_stat, t_p = paired_ttest(errors1, errors2)
        
        # Wilcoxon signed-rank
        w_stat, w_p = wilcoxon_test(errors1, errors2)
        
        # Mean difference with CI
        diff = np.abs(errors1) - np.abs(errors2)
        mean_diff = float(np.mean(diff))
        se = float(np.std(diff) / np.sqrt(len(diff)))
        ci = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)
        
        return ComparisonResult(
            model1_name=self.config.experiment_name,
            model2_name=other_name,
            dm_statistic=dm_stat,
            dm_pvalue=dm_p,
            ttest_statistic=t_stat,
            ttest_pvalue=t_p,
            wilcoxon_statistic=w_stat,
            wilcoxon_pvalue=w_p,
            mean_diff=mean_diff,
            mean_diff_ci=ci,
            significant_dm=dm_p < significance_level,
            significant_ttest=t_p < significance_level,
            significant_wilcoxon=w_p < significance_level
        )


# ============================================================================
# Visualization Class
# ============================================================================

class ResultVisualizer:
    """
    Publication-quality visualization generator.
    
    Creates figures suitable for academic publication with proper
    formatting, annotations, and statistical information.
    """
    
    def __init__(
        self,
        config,
        output_dir: str = None,
        use_latex: bool = False,
        palette: str = 'default'
    ):
        """
        Initialize visualizer.
        
        Args:
            config: Experiment configuration
            output_dir: Directory for saving figures
            use_latex: Use LaTeX for text rendering
            palette: Color palette name
        """
        self.config = config
        self.output_dir = output_dir or os.path.join(
            getattr(config, 'results_dir', './results'), 'figures'
        )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.use_latex = use_latex
        self.colors = ColorPalette.get_colors(10, palette)
        
        # Setup matplotlib style
        if MATPLOTLIB_AVAILABLE:
            setup_matplotlib_style(use_latex)
    
    def _save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        formats: List[str] = ['pdf', 'png']
    ) -> List[str]:
        """Save figure in multiple formats."""
        paths = []
        for fmt in formats:
            filepath = os.path.join(self.output_dir, f"{filename}.{fmt}")
            fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
            paths.append(filepath)
        plt.close(fig)
        return paths
    
    def plot_prediction_vs_actual(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        title: str = "Predicted vs Actual PV Output",
        filename: str = "pred_vs_actual"
    ) -> List[str]:
        """
        Create scatter plot of predictions vs actual values.
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Scatter plot
        if uncertainties is not None:
            scatter = ax.scatter(
                targets, predictions, c=uncertainties,
                cmap='viridis', alpha=0.6, s=20, edgecolors='none'
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Uncertainty (kW)')
        else:
            ax.scatter(
                targets, predictions, alpha=0.5, s=20,
                c=self.colors[0], edgecolors='none'
            )
        
        # Perfect prediction line
        lims = [
            min(targets.min(), predictions.min()),
            max(targets.max(), predictions.max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, linewidth=1.5, label='Perfect forecast')
        
        # Regression line
        if SCIPY_AVAILABLE:
            slope, intercept, r_value, _, _ = stats.linregress(targets, predictions)
            x_line = np.linspace(lims[0], lims[1], 100)
            ax.plot(
                x_line, slope * x_line + intercept,
                color=self.colors[1], linewidth=1.5,
                label=f'Fit ($R^2$ = {r_value**2:.3f})'
            )
        
        # Labels
        ax.set_xlabel('Actual PV Output (kW)')
        ax.set_ylabel('Predicted PV Output (kW)')
        ax.set_title(title)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        
        # Statistics annotation
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        stats_text = f'MAE: {mae:.2f} kW\nRMSE: {rmse:.2f} kW'
        ax.text(
            0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        )
        
        return self._save_figure(fig, filename)
    
    def plot_time_series(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        n_points: int = 200,
        title: str = "PV Forecast Time Series",
        filename: str = "time_series"
    ) -> List[str]:
        """
        Create time series plot with prediction intervals.
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        n = min(n_points, len(predictions))
        idx = np.arange(n)
        
        # Plot prediction interval
        ax.fill_between(
            idx, lower[:n], upper[:n],
            alpha=0.3, color=self.colors[0],
            label='90% Prediction Interval'
        )
        
        # Plot predictions and actual
        ax.plot(
            idx, predictions[:n], color=self.colors[0],
            linewidth=1.5, label='Prediction'
        )
        ax.plot(
            idx, targets[:n], color=self.colors[1],
            linewidth=1.5, label='Actual', alpha=0.8
        )
        
        # Labels
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('PV Output (kW)')
        ax.set_title(title)
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Coverage annotation
        covered = (targets[:n] >= lower[:n]) & (targets[:n] <= upper[:n])
        coverage = np.mean(covered)
        ax.text(
            0.02, 0.98, f'Coverage: {100*coverage:.1f}%',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )
        
        return self._save_figure(fig, filename)
    
    def plot_reliability_diagram(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        title: str = "Reliability Diagram",
        filename: str = "reliability"
    ) -> List[str]:
        """
        Create reliability diagram for calibration analysis.
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        rel_data = compute_reliability_diagram(predictions, uncertainties, targets)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        
        # Left: Reliability diagram
        ax1 = axes[0]
        ax1.plot(
            rel_data.expected_coverage, rel_data.empirical_coverage,
            'o-', color=self.colors[0], linewidth=2, markersize=8,
            label='Model'
        )
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
        ax1.fill_between(
            rel_data.expected_coverage,
            rel_data.expected_coverage,
            rel_data.empirical_coverage,
            alpha=0.2, color=self.colors[1]
        )
        
        ax1.set_xlabel('Expected Coverage')
        ax1.set_ylabel('Empirical Coverage')
        ax1.set_title('Reliability Diagram')
        ax1.legend(loc='lower right', framealpha=0.9)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_aspect('equal')
        
        ax1.text(
            0.05, 0.95, f'Cal. Error: {rel_data.calibration_error:.3f}',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )
        
        # Right: PIT histogram
        ax2 = axes[1]
        if rel_data.pit_values is not None:
            ax2.hist(
                rel_data.pit_values, bins=20, density=True,
                alpha=0.7, color=self.colors[0], edgecolor='white'
            )
            ax2.axhline(1.0, color='k', linestyle='--', linewidth=1.5, label='Uniform')
            ax2.set_xlabel('PIT Value')
            ax2.set_ylabel('Density')
            ax2.set_title('PIT Histogram')
            ax2.legend(loc='upper right')
            ax2.set_xlim([0, 1])
        
        return self._save_figure(fig, filename)
    
    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        title: str = "Forecast Error Distribution",
        filename: str = "error_dist"
    ) -> List[str]:
        """
        Create error distribution histogram with fitted normal.
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        errors = predictions - targets
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Histogram
        ax.hist(
            errors, bins=50, density=True, alpha=0.7,
            color=self.colors[0], edgecolor='white', label='Empirical'
        )
        
        # Fitted normal
        if SCIPY_AVAILABLE:
            mu, std = stats.norm.fit(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            ax.plot(
                x, stats.norm.pdf(x, mu, std), color=self.colors[1],
                linewidth=2, label=f'Normal ($\\mu$={mu:.2f}, $\\sigma$={std:.2f})'
            )
        
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Forecast Error (kW)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Statistics
        bias = np.mean(errors)
        mae = np.mean(np.abs(errors))
        ax.text(
            0.02, 0.98, f'Bias: {bias:.3f} kW\nMAE: {mae:.3f} kW',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )
        
        return self._save_figure(fig, filename)
    
    def plot_spread_skill(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10,
        title: str = "Spread-Skill Diagram",
        filename: str = "spread_skill"
    ) -> List[str]:
        """
        Create spread-skill diagram.
        
        Shows relationship between predicted uncertainty (spread)
        and actual forecast error (skill).
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        errors = np.abs(predictions - targets)
        
        # Bin by predicted uncertainty
        percentiles = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(uncertainties, percentiles[1:-1])
        
        mean_spread = []
        mean_error = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                mean_spread.append(np.mean(uncertainties[mask]))
                mean_error.append(np.mean(errors[mask]))
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        ax.scatter(mean_spread, mean_error, s=60, c=self.colors[0])
        
        # Perfect calibration line
        lims = [0, max(max(mean_spread), max(mean_error)) * 1.1]
        ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect calibration')
        
        ax.set_xlabel('Mean Predicted Std (kW)')
        ax.set_ylabel('Mean Absolute Error (kW)')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        
        return self._save_figure(fig, filename)
    
    def plot_ablation_comparison(
        self,
        results: Dict[str, EvaluationResults],
        metrics: List[str] = None,
        title: str = "Ablation Study Results",
        filename: str = "ablation"
    ) -> List[str]:
        """
        Create bar chart comparing ablation study results.
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        if metrics is None:
            metrics = ['mae', 'rmse', 'crps', 'coverage_90']
        
        model_names = list(results.keys())
        n_models = len(model_names)
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        colors = ColorPalette.get_colors(n_models)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            values = []
            errors = []
            for name in model_names:
                attr = getattr(results[name], metric, None)
                if attr is None:
                    values.append(0)
                    errors.append(0)
                elif isinstance(attr, MetricResult):
                    values.append(attr.value)
                    if attr.std_error:
                        errors.append(attr.std_error)
                    else:
                        errors.append(0)
                else:
                    values.append(attr)
                    errors.append(0)
            
            bars = ax.bar(range(n_models), values, color=colors, yerr=errors, capsize=3)
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            
            # Format metric name
            metric_label = metric.upper().replace('_', ' ')
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8
                )
        
        fig.suptitle(title, fontsize=12, y=1.02)
        
        return self._save_figure(fig, filename)
    
    def create_comprehensive_figure(
        self,
        evaluator: ModelEvaluator,
        results: EvaluationResults,
        title: str = "SkyGPT-ViTODE Evaluation",
        filename: str = "comprehensive"
    ) -> List[str]:
        """
        Create multi-panel comprehensive evaluation figure.
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        pred = evaluator.predictions
        target = evaluator.targets
        unc = evaluator.uncertainties
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Panel A: Prediction vs Actual
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(target, pred, alpha=0.4, s=8, c=self.colors[0])
        lims = [min(target.min(), pred.min()), max(target.max(), pred.max())]
        ax1.plot(lims, lims, 'k--', alpha=0.75, linewidth=1.5)
        ax1.set_xlabel('Actual (kW)')
        ax1.set_ylabel('Predicted (kW)')
        ax1.set_title('(a) Predicted vs Actual')
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_aspect('equal')
        
        # Panel B: Error distribution
        ax2 = fig.add_subplot(gs[0, 1])
        errors = pred - target
        ax2.hist(errors, bins=40, density=True, alpha=0.7, color=self.colors[0])
        if SCIPY_AVAILABLE:
            mu, std = stats.norm.fit(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            ax2.plot(x, stats.norm.pdf(x, mu, std), color=self.colors[1], linewidth=2)
        ax2.axvline(0, color='k', linestyle='--')
        ax2.set_xlabel('Error (kW)')
        ax2.set_ylabel('Density')
        ax2.set_title('(b) Error Distribution')
        
        # Panel C: Reliability diagram
        ax3 = fig.add_subplot(gs[0, 2])
        rel_data = compute_reliability_diagram(pred, unc, target)
        ax3.plot(
            rel_data.expected_coverage, rel_data.empirical_coverage,
            'o-', color=self.colors[0], linewidth=2, markersize=6
        )
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
        ax3.set_xlabel('Expected Coverage')
        ax3.set_ylabel('Empirical Coverage')
        ax3.set_title('(c) Reliability Diagram')
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        ax3.set_aspect('equal')
        
        # Panel D: PIT histogram
        ax4 = fig.add_subplot(gs[1, 0])
        if rel_data.pit_values is not None:
            ax4.hist(rel_data.pit_values, bins=20, density=True, alpha=0.7, color=self.colors[0])
            ax4.axhline(1.0, color='k', linestyle='--', linewidth=1.5)
        ax4.set_xlabel('PIT Value')
        ax4.set_ylabel('Density')
        ax4.set_title('(d) PIT Histogram')
        ax4.set_xlim([0, 1])
        
        # Panel E: Time series
        ax5 = fig.add_subplot(gs[1, 1:])
        n = min(150, len(pred))
        idx = np.arange(n)
        q90 = GAUSSIAN_QUANTILES[0.90]
        lower = pred[:n] - q90 * unc[:n]
        upper = pred[:n] + q90 * unc[:n]
        ax5.fill_between(idx, lower, upper, alpha=0.3, color=self.colors[0])
        ax5.plot(idx, pred[:n], color=self.colors[0], linewidth=1.5, label='Prediction')
        ax5.plot(idx, target[:n], color=self.colors[1], linewidth=1.5, label='Actual')
        ax5.set_xlabel('Sample Index')
        ax5.set_ylabel('PV Output (kW)')
        ax5.set_title('(e) Time Series with 90% Prediction Interval')
        ax5.legend(loc='upper right')
        
        # Panel F: Metrics summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        metrics_text = (
            f"{'='*60}\n"
            f"EVALUATION SUMMARY\n"
            f"{'='*60}\n\n"
            f"Point Prediction Metrics:\n"
            f"  MAE:           {results.mae.value:.4f} kW\n"
            f"  RMSE:          {results.rmse.value:.4f} kW\n"
            f"  MBE:           {results.mbe.value:.4f} kW\n"
            f"  R²:            {results.r2.value:.4f}\n\n"
            f"Probabilistic Metrics:\n"
            f"  CRPS:          {results.crps.value:.4f} kW\n"
            f"  Winkler (90%): {results.winkler_90.value:.4f}\n"
            f"  Coverage (90%): {100*results.coverage_90.value:.1f}%\n"
            f"  Interval Width: {results.avg_interval_width_90.value:.4f} kW\n"
            f"  Cal. Error:    {results.calibration_error.value:.4f}\n\n"
            f"N samples: {results.n_samples}"
        )
        
        ax6.text(
            0.5, 0.5, metrics_text, transform=ax6.transAxes,
            fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray')
        )
        ax6.set_title('(f) Summary Metrics')
        
        fig.suptitle(title, fontsize=14, y=0.98)
        
        return self._save_figure(fig, filename)


# ============================================================================
# Results Export Functions
# ============================================================================

def export_results_to_latex(
    results: Dict[str, EvaluationResults],
    output_path: str,
    metrics: List[str] = None,
    caption: str = "Comparison of Solar Forecasting Methods",
    label: str = "tab:results"
) -> str:
    """
    Export evaluation results to LaTeX table with best-value highlighting.
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'crps', 'winkler_90', 'coverage_90']
    
    # Find best values for highlighting
    best_values = {}
    for metric in metrics:
        values = []
        for res in results.values():
            attr = getattr(res, metric, None)
            if attr is not None:
                val = attr.value if isinstance(attr, MetricResult) else attr
                values.append(val)
        
        if values:
            # For coverage, closest to nominal is best
            if 'coverage' in metric:
                nominal = float(metric.split('_')[1]) / 100
                best_idx = np.argmin([abs(v - nominal) for v in values])
                best_values[metric] = values[best_idx]
            # For error metrics, lower is better
            else:
                best_values[metric] = min(values)
    
    # Header
    metric_headers = [m.upper().replace('_', ' ') for m in metrics]
    
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Model & " + " & ".join(metric_headers) + r" \\",
        r"\midrule",
    ]
    
    for name, res in results.items():
        row = res.to_latex_row(precision=4, metrics=metrics, best_values=best_values)
        latex.append(row)
    
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    content = "\n".join(latex)
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    logger.info(f"LaTeX table saved to: {output_path}")
    return output_path


def export_results_to_csv(
    results: Dict[str, EvaluationResults],
    output_path: str
) -> str:
    """Export evaluation results to CSV."""
    if not PANDAS_AVAILABLE:
        # Fallback to manual CSV
        rows = [res.to_flat_dict() for res in results.values()]
        
        with open(output_path, 'w') as f:
            if rows:
                headers = list(rows[0].keys())
                f.write(','.join(headers) + '\n')
                for row in rows:
                    f.write(','.join(str(row.get(h, '')) for h in headers) + '\n')
        
        return output_path
    
    rows = []
    for name, res in results.items():
        row = res.to_flat_dict()
        row['model_name'] = name
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    logger.info(f"CSV saved to: {output_path}")
    return output_path


def export_results_to_json(
    results: Dict[str, EvaluationResults],
    output_path: str
) -> str:
    """Export evaluation results to JSON."""
    data = {name: res.to_dict() for name, res in results.items()}
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"JSON saved to: {output_path}")
    return output_path


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def evaluate_model(
    checkpoint_path: str,
    config,
    output_dir: Optional[str] = None,
    n_samples: int = 50,
    compute_bootstrap: bool = True,
    save_figures: bool = True,
    save_predictions: bool = True
) -> EvaluationResults:
    """
    Complete evaluation pipeline for a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Experiment configuration
        output_dir: Directory for saving results
        n_samples: Number of samples for probabilistic prediction
        compute_bootstrap: Compute bootstrap confidence intervals
        save_figures: Generate and save figures
        save_predictions: Save raw predictions
        
    Returns:
        EvaluationResults object
    """
    logger.info("=" * 60)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 60)
    
    output_dir = output_dir or getattr(config, 'results_dir', './results')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from: {checkpoint_path}")
    
    from models import create_model
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from Lightning)
        state_dict = {
            k.replace('model.', ''): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Create data module
    from data import create_datamodule
    datamodule = create_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup('test')
    
    test_loader = datamodule.test_dataloader()
    logger.info(f"Test samples: {datamodule.n_test_samples}")
    
    # Create evaluator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator(model, config, device)
    
    # Run inference
    evaluator.run_inference(test_loader, n_samples=n_samples)
    
    # Calibrate conformal predictor
    coverage_levels = getattr(config.conformal, 'coverage_levels', [0.80, 0.90, 0.95])
    calibration_ratio = getattr(config.conformal, 'calibration_ratio', 0.2)
    calibrated_quantiles = evaluator.calibrate_conformal(calibration_ratio, coverage_levels)
    
    # Compute metrics
    results = evaluator.compute_metrics(
        calibrated_quantiles=calibrated_quantiles,
        model_name=config.experiment_name,
        compute_bootstrap=compute_bootstrap
    )
    
    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(f"  MAE:        {results.mae}")
    logger.info(f"  RMSE:       {results.rmse}")
    logger.info(f"  CRPS:       {results.crps}")
    logger.info(f"  Winkler90:  {results.winkler_90}")
    logger.info(f"  Coverage90: {100*results.coverage_90.value:.1f}%")
    logger.info(f"  Cal Error:  {results.calibration_error}")
    
    # Generate visualizations
    if save_figures and MATPLOTLIB_AVAILABLE:
        logger.info("\nGenerating figures...")
        visualizer = ResultVisualizer(config, os.path.join(output_dir, 'figures'))
        
        visualizer.create_comprehensive_figure(
            evaluator, results,
            title=f"{config.experiment_name} Evaluation"
        )
        
        visualizer.plot_prediction_vs_actual(
            evaluator.predictions, evaluator.targets, evaluator.uncertainties
        )
        
        visualizer.plot_reliability_diagram(
            evaluator.predictions, evaluator.uncertainties, evaluator.targets
        )
        
        visualizer.plot_error_distribution(
            evaluator.predictions, evaluator.targets
        )
        
        visualizer.plot_spread_skill(
            evaluator.predictions, evaluator.uncertainties, evaluator.targets
        )
        
        # Time series
        q90 = calibrated_quantiles.get(0.90, 1.645)
        lower = evaluator.predictions - q90 * evaluator.uncertainties
        upper = evaluator.predictions + q90 * evaluator.uncertainties
        visualizer.plot_time_series(
            evaluator.predictions, evaluator.targets, lower, upper
        )
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    export_results_to_json({config.experiment_name: results}, results_path)
    
    # Save predictions
    if save_predictions:
        pred_path = os.path.join(output_dir, 'predictions.npz')
        np.savez(
            pred_path,
            predictions=evaluator.predictions,
            uncertainties=evaluator.uncertainties,
            targets=evaluator.targets,
            samples=evaluator.samples
        )
        logger.info(f"Predictions saved to: {pred_path}")
    
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)
    
    return results


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SkyGPT-ViTODE Evaluation Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help="Path to SKIPP'D dataset"
    )
    parser.add_argument(
        '--n-samples', type=int, default=50,
        help='Number of samples for probabilistic prediction'
    )
    parser.add_argument(
        '--no-bootstrap', action='store_true',
        help='Skip bootstrap confidence intervals'
    )
    parser.add_argument(
        '--no-figures', action='store_true',
        help='Skip figure generation'
    )
    parser.add_argument(
        '--no-save-predictions', action='store_true',
        help='Do not save raw predictions'
    )
    parser.add_argument(
        '--use-latex', action='store_true',
        help='Use LaTeX for figure text rendering'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load configuration
    from config import ExperimentConfig
    
    if args.config is not None:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig()
    
    # Override with command line arguments
    if args.data_path is not None:
        config.data.data_path = args.data_path
    if args.output_dir is not None:
        config.results_dir = args.output_dir
    
    # Setup matplotlib
    if MATPLOTLIB_AVAILABLE:
        setup_matplotlib_style(use_latex=args.use_latex)
    
    # Run evaluation
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        config=config,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        compute_bootstrap=not args.no_bootstrap,
        save_figures=not args.no_figures,
        save_predictions=not args.no_save_predictions
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {config.experiment_name}")
    print(f"Test samples: {results.n_samples}")
    print("-" * 60)
    print(f"MAE:            {results.mae}")
    print(f"RMSE:           {results.rmse}")
    print(f"CRPS:           {results.crps}")
    print(f"Winkler (90%):  {results.winkler_90}")
    print(f"Coverage (90%): {100*results.coverage_90.value:.1f}%")
    print(f"Interval Width: {results.avg_interval_width_90}")
    print(f"Cal. Error:     {results.calibration_error}")
    print("=" * 60)


if __name__ == "__main__":
    main()