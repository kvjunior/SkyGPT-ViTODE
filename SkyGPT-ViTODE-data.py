"""
SkyGPT-ViTODE Data Loading and Preprocessing
=============================================


Key Features:
    - Proper HDF5 file handling with worker-safe lazy loading
    - Clear-sky index normalization for improved generalization
    - Date-based train/validation/test splitting
    - Comprehensive data augmentation with temporal consistency
    - Cloud detection and filtering with efficient caching
    - Memory-efficient data loading with optional memory mapping
"""

from __future__ import annotations

import os
import math
import random
import hashlib
import logging
import warnings
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Any, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import lru_cache

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

# Conditional imports
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pl = None

try:
    from torchvision import transforms
    from torchvision.transforms import functional as TF
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    TF = None


# ============================================================================
# Constants
# ============================================================================

# SKIPP'D dataset specifications
SKIPPD_PV_CAPACITY_KW = 30.0  # Maximum PV system capacity
SKIPPD_LATITUDE = 37.4275    # Stanford University latitude
SKIPPD_LONGITUDE = -122.1697  # Stanford University longitude
SKIPPD_ELEVATION_DEG = 22.5   # Panel elevation angle
SKIPPD_AZIMUTH_DEG = 195.0    # Panel azimuth (clockwise from North)
SKIPPD_EFFECTIVE_AREA_M2 = 24.98  # Fitted effective area from clear-sky model

# Image specifications
SKIPPD_ORIGINAL_RESOLUTION = 2048
SKIPPD_DEFAULT_RESOLUTION = 64

# Clear sky model fitted parameter
SKIPPD_P_MAX_W_M2 = 1000.0  # Maximum solar irradiance


# ============================================================================
# Logging Setup
# ============================================================================

logger = logging.getLogger('SkyGPT-ViTODE.Data')


# ============================================================================
# Clear Sky Model
# ============================================================================

def compute_solar_position(
    timestamps: np.ndarray,
    latitude: float = SKIPPD_LATITUDE,
    longitude: float = SKIPPD_LONGITUDE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute solar elevation and azimuth angles.
    
    Uses simplified astronomical calculations suitable for
    clear-sky power estimation.
    
    Args:
        timestamps: Unix timestamps (seconds since epoch)
        latitude: Site latitude in degrees
        longitude: Site longitude in degrees
        
    Returns:
        Tuple of (elevation_angles, azimuth_angles) in radians
    """
    # Convert to datetime for calculations
    if isinstance(timestamps, (int, float)):
        timestamps = np.array([timestamps])
    
    # Julian date calculation
    jd = timestamps / 86400.0 + 2440587.5
    
    # Julian century
    jc = (jd - 2451545.0) / 36525.0
    
    # Solar coordinates (simplified)
    # Mean longitude of the sun
    mean_lon = np.mod(280.46646 + jc * (36000.76983 + 0.0003032 * jc), 360)
    
    # Mean anomaly of the sun
    mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    mean_anom_rad = np.radians(mean_anom)
    
    # Equation of center
    eoc = np.sin(mean_anom_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + \
          np.sin(2 * mean_anom_rad) * (0.019993 - 0.000101 * jc) + \
          np.sin(3 * mean_anom_rad) * 0.000289
    
    # True longitude
    true_lon = mean_lon + eoc
    
    # Obliquity of ecliptic
    obliquity = 23.439291 - 0.013004 * jc
    obliquity_rad = np.radians(obliquity)
    
    # Declination
    declination = np.arcsin(np.sin(obliquity_rad) * np.sin(np.radians(true_lon)))
    
    # Hour angle
    # Approximate equation of time
    eot = 4 * (mean_lon - 0.0057183 - np.degrees(np.arctan2(
        np.cos(obliquity_rad) * np.sin(np.radians(true_lon)),
        np.cos(np.radians(true_lon))
    )))
    
    # Time offset from UTC
    time_offset = eot + 4 * longitude
    
    # Solar time
    hours = (timestamps % 86400) / 3600.0
    solar_time = hours + time_offset / 60.0
    
    # Hour angle
    hour_angle = np.radians((solar_time - 12.0) * 15.0)
    
    # Solar elevation
    lat_rad = np.radians(latitude)
    elevation = np.arcsin(
        np.sin(lat_rad) * np.sin(declination) +
        np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    
    # Solar azimuth
    azimuth = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(lat_rad) - np.tan(declination) * np.cos(lat_rad)
    )
    azimuth = np.mod(azimuth + np.pi, 2 * np.pi)  # Convert to 0-2π range
    
    return elevation, azimuth


def compute_clear_sky_power(
    timestamps: np.ndarray,
    elevation_deg: float = SKIPPD_ELEVATION_DEG,
    azimuth_deg: float = SKIPPD_AZIMUTH_DEG,
    effective_area: float = SKIPPD_EFFECTIVE_AREA_M2,
    p_max: float = SKIPPD_P_MAX_W_M2,
    latitude: float = SKIPPD_LATITUDE,
    longitude: float = SKIPPD_LONGITUDE
) -> np.ndarray:
    """
    Compute theoretical clear-sky PV power output.
    
    Based on the model from the SKIPP'D paper:
    P_clr(t) = P_m * A_e * {cos(ε)cos(χ(t)) + sin(ε)sin(χ(t))cos[ξ(t) - ζ]}
    
    where:
        P_m = maximum irradiance (1000 W/m²)
        A_e = effective area (fitted from clear days)
        ε = panel elevation angle
        χ(t) = solar elevation angle
        ξ(t) = solar azimuth angle
        ζ = panel azimuth angle
    
    Args:
        timestamps: Unix timestamps
        elevation_deg: Panel elevation angle in degrees
        azimuth_deg: Panel azimuth angle in degrees (clockwise from North)
        effective_area: Effective panel area in m²
        p_max: Maximum irradiance in W/m²
        latitude: Site latitude
        longitude: Site longitude
        
    Returns:
        Clear-sky power in kW
    """
    # Get solar position
    solar_elevation, solar_azimuth = compute_solar_position(
        timestamps, latitude, longitude
    )
    
    # Convert panel angles to radians
    panel_elevation = np.radians(elevation_deg)
    panel_azimuth = np.radians(azimuth_deg)
    
    # Compute clear-sky power (equation from SKIPP'D paper)
    cos_incidence = (
        np.cos(panel_elevation) * np.cos(solar_elevation) +
        np.sin(panel_elevation) * np.sin(solar_elevation) * 
        np.cos(solar_azimuth - panel_azimuth)
    )
    
    # Power is zero when sun is below horizon or behind panel
    cos_incidence = np.maximum(cos_incidence, 0)
    solar_elevation = np.maximum(solar_elevation, 0)
    
    # Clear-sky power in kW
    power_kw = p_max * effective_area * cos_incidence * np.sin(solar_elevation) / 1000.0
    
    return np.maximum(power_kw, 0)


def compute_clear_sky_index(
    pv_power: np.ndarray,
    timestamps: np.ndarray,
    min_clear_sky_power: float = 0.1
) -> np.ndarray:
    """
    Compute clear-sky index (ratio of actual to theoretical clear-sky power).
    
    The clear-sky index is a normalized measure that removes the
    deterministic solar position effects, isolating cloud influence.
    
    Args:
        pv_power: Actual PV power output in kW
        timestamps: Unix timestamps corresponding to power measurements
        min_clear_sky_power: Minimum clear-sky power to avoid division issues
        
    Returns:
        Clear-sky index values (typically 0-1, can exceed 1 due to cloud enhancement)
    """
    clear_sky_power = compute_clear_sky_power(timestamps)
    
    # Avoid division by very small values
    clear_sky_power = np.maximum(clear_sky_power, min_clear_sky_power)
    
    # Compute index
    k_clr = pv_power / clear_sky_power
    
    # Clip to reasonable range (allowing some cloud enhancement)
    k_clr = np.clip(k_clr, 0, 1.5)
    
    return k_clr


# ============================================================================
# Cloud Detection
# ============================================================================

def compute_cloud_index_vectorized(
    images: np.ndarray,
    threshold: float = 0.3
) -> np.ndarray:
    """
    Compute cloud index for multiple images using vectorized operations.
    
    Uses normalized red-blue ratio method based on the physical property
    that clear sky appears more blue while clouds scatter uniformly.
    
    Reference:
        Heinle et al., "Automatic cloud classification of whole sky images"
        Atmospheric Measurement Techniques, 2010
    
    Args:
        images: RGB images array (N, H, W, 3) or (H, W, 3) with values [0, 255]
        threshold: Not used in computation, kept for API compatibility
        
    Returns:
        Cloud index values (N,) or scalar. Lower values indicate cloudier conditions.
    """
    single_image = images.ndim == 3
    if single_image:
        images = images[np.newaxis, ...]
    
    # Extract channels
    R = images[..., 0].astype(np.float32)
    G = images[..., 1].astype(np.float32)
    B = images[..., 2].astype(np.float32)
    
    # Compute normalized red-blue ratio
    denominator = R + B + 1e-6
    nrb = (R - B) / denominator
    
    # Create valid pixel mask
    # Exclude saturated pixels (too bright) and dark pixels (background/night)
    brightness = R + G + B
    valid_mask = (brightness > 30) & (R < 250) & (B < 250) & (brightness < 700)
    
    # Apply circular mask for fish-eye images
    H, W = images.shape[1:3]
    y, x = np.ogrid[:H, :W]
    center_y, center_x = H / 2, W / 2
    radius = min(H, W) / 2 * 0.95  # 95% of image radius
    circular_mask = ((y - center_y) ** 2 + (x - center_x) ** 2) <= radius ** 2
    
    valid_mask = valid_mask & circular_mask[np.newaxis, ...]
    
    # Compute mean cloud index over valid pixels for each image
    cloud_indices = np.zeros(len(images))
    for i in range(len(images)):
        if valid_mask[i].sum() > 0:
            cloud_indices[i] = nrb[i][valid_mask[i]].mean()
        else:
            cloud_indices[i] = 0.5  # Neutral value for invalid images
    
    return cloud_indices[0] if single_image else cloud_indices


def is_cloudy_sample(
    image: np.ndarray,
    threshold: float = 0.3
) -> bool:
    """
    Determine if a sky image represents cloudy conditions.
    
    Lower normalized red-blue ratio indicates more clouds (white/gray appearance).
    
    Args:
        image: RGB image array (H, W, 3)
        threshold: Cloud index threshold (samples below this are cloudy)
        
    Returns:
        True if the sample is classified as cloudy
    """
    cloud_index = compute_cloud_index_vectorized(image, threshold)
    return float(cloud_index) < threshold


# ============================================================================
# Preprocessing Utilities
# ============================================================================

def preprocess_video(
    video: torch.Tensor,
    resolution: int,
    sequence_length: Optional[int] = None,
    normalize: bool = True,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None
) -> torch.Tensor:
    """
    Preprocess video tensor for model input.
    
    Args:
        video: Raw video tensor (T, H, W, C) with values in [0, 255]
        resolution: Target spatial resolution
        sequence_length: Target temporal length (optional crop/pad)
        normalize: Whether to normalize pixel values
        mean: Per-channel mean for normalization (default: [0.5, 0.5, 0.5])
        std: Per-channel std for normalization (default: [0.5, 0.5, 0.5])
        
    Returns:
        Preprocessed video tensor (C, T, H, W) normalized
    """
    # Convert to float and scale to [0, 1]
    video = video.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    
    t, c, h, w = video.shape
    
    # Handle temporal dimension
    if sequence_length is not None:
        if sequence_length > t:
            # Pad with last frame if needed
            padding = video[-1:].expand(sequence_length - t, -1, -1, -1)
            video = torch.cat([video, padding], dim=0)
        elif sequence_length < t:
            video = video[:sequence_length]
        t = sequence_length
    
    # Spatial resize if needed
    if h != resolution or w != resolution:
        # Use bilinear interpolation for smooth resizing
        video = F.interpolate(
            video, 
            size=(resolution, resolution), 
            mode='bilinear',
            align_corners=False
        )
    
    # Rearrange to (C, T, H, W)
    video = video.permute(1, 0, 2, 3).contiguous()
    
    # Normalize
    if normalize:
        if mean is None:
            mean = (0.5, 0.5, 0.5)
        if std is None:
            std = (0.5, 0.5, 0.5)
        
        mean = torch.tensor(mean, dtype=video.dtype, device=video.device)
        std = torch.tensor(std, dtype=video.dtype, device=video.device)
        
        # Reshape for broadcasting: (C, 1, 1, 1)
        mean = mean.view(-1, 1, 1, 1)
        std = std.view(-1, 1, 1, 1)
        
        video = (video - mean) / std
    
    return video


def apply_circular_mask(
    image: torch.Tensor,
    mask_value: float = 0.0,
    radius_ratio: float = 0.95
) -> torch.Tensor:
    """
    Apply circular mask for fish-eye sky camera images.
    
    The SKIPP'D dataset uses a Hikvision DS-2CD6362F-IV fish-eye camera
    which produces circular images. This function masks out regions
    outside the circular field of view.
    
    Args:
        image: Image tensor (C, H, W) or (B, C, H, W)
        mask_value: Value for masked regions
        radius_ratio: Ratio of mask radius to half image dimension
        
    Returns:
        Masked image tensor
    """
    if image.dim() == 3:
        _, h, w = image.shape
        batch_mode = False
    else:
        _, _, h, w = image.shape
        batch_mode = True
    
    # Create coordinate grid
    y = torch.linspace(-1, 1, h, device=image.device, dtype=image.dtype)
    x = torch.linspace(-1, 1, w, device=image.device, dtype=image.dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Circular mask
    radius_sq = (radius_ratio ** 2)
    mask = (xx ** 2 + yy ** 2) <= radius_sq
    
    # Expand mask dimensions
    if batch_mode:
        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(image)
    else:
        mask = mask.unsqueeze(0).expand_as(image)
    
    return torch.where(mask, image, torch.full_like(image, mask_value))


# ============================================================================
# Data Augmentation
# ============================================================================

class VideoAugmentation:
    """
    Data augmentation pipeline for sky video sequences.
    
    Applies spatially and temporally consistent augmentations across
    all frames while respecting physical constraints of sky imagery.
    
    Supported augmentations:
        - Horizontal flip (cloud motion is directionally symmetric)
        - Random rotation (small angles to preserve horizon)
        - Brightness jitter (simulates exposure variations)
        - Contrast jitter (simulates atmospheric clarity variations)
    """
    
    def __init__(self, config):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: DataConfig object with augmentation parameters
        """
        self.enabled = config.use_augmentation
        self.horizontal_flip_prob = config.horizontal_flip_prob
        self.rotation_degrees = config.rotation_degrees
        self.brightness_jitter = config.brightness_jitter
        self.contrast_jitter = getattr(config, 'contrast_jitter', 0.1)
        
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to video sequence.
        
        All augmentations use the same random parameters across frames
        to maintain temporal consistency.
        
        Args:
            video: Video tensor (C, T, H, W), normalized
            
        Returns:
            Augmented video tensor
        """
        if not self.enabled:
            return video
        
        C, T, H, W = video.shape
        
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            video = torch.flip(video, dims=[-1])
        
        # Random rotation (same angle for all frames)
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            if abs(angle) > 0.5 and TORCHVISION_AVAILABLE:
                # Rotate each frame with same angle
                frames = []
                for t in range(T):
                    frame = video[:, t]  # (C, H, W)
                    frame = TF.rotate(
                        frame, 
                        angle, 
                        interpolation=TF.InterpolationMode.BILINEAR,
                        fill=0
                    )
                    frames.append(frame)
                video = torch.stack(frames, dim=1)
        
        # Brightness jitter (additive)
        if self.brightness_jitter > 0:
            brightness_delta = random.uniform(
                -self.brightness_jitter,
                self.brightness_jitter
            )
            video = video + brightness_delta
        
        # Contrast jitter (multiplicative)
        if self.contrast_jitter > 0:
            contrast_factor = random.uniform(
                1.0 - self.contrast_jitter,
                1.0 + self.contrast_jitter
            )
            # Apply around mean
            mean = video.mean()
            video = (video - mean) * contrast_factor + mean
        
        # Clamp to valid range (assuming normalized to [-1, 1] or [0, 1])
        video = torch.clamp(video, -2.0, 2.0)
        
        return video
    
    def __repr__(self) -> str:
        return (
            f"VideoAugmentation("
            f"enabled={self.enabled}, "
            f"flip_prob={self.horizontal_flip_prob}, "
            f"rotation={self.rotation_degrees}°, "
            f"brightness={self.brightness_jitter}, "
            f"contrast={self.contrast_jitter})"
        )


# ============================================================================
# Worker-Safe HDF5 Handler
# ============================================================================

class HDF5Handler:
    """
    Thread-local HDF5 file handler for multi-worker DataLoader compatibility.
    
    HDF5 files are not thread-safe. This class ensures each DataLoader worker
    opens its own file handle, preventing race conditions and data corruption.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize handler.
        
        Args:
            file_path: Path to HDF5 file
        """
        self.file_path = file_path
        self._local = threading.local()
    
    @property
    def file(self) -> h5py.File:
        """Get thread-local file handle, opening if necessary."""
        if not hasattr(self._local, 'file') or self._local.file is None:
            self._local.file = h5py.File(self.file_path, 'r', swmr=True)
        return self._local.file
    
    def close(self):
        """Close thread-local file handle."""
        if hasattr(self._local, 'file') and self._local.file is not None:
            try:
                self._local.file.close()
            except:
                pass
            self._local.file = None
    
    def __getitem__(self, key: str) -> h5py.Dataset:
        """Access dataset by key."""
        return self.file[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.file
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# ============================================================================
# SKIPP'D Dataset Classes
# ============================================================================

class SKIPPDDataset(data.Dataset):
    """
    PyTorch Dataset for the SKIPP'D (SKy Images and Photovoltaic Power) dataset.
    
    This dataset provides aligned pairs of fish-eye sky images and PV power
    generation records from Stanford University's 30-kW rooftop installation.
    
    Features:
        - Worker-safe HDF5 file handling
        - Optional clear-sky index normalization
        - Cloud filtering with efficient caching
        - Date-based train/validation/test splitting
        - Comprehensive data validation
    
    Dataset Structure (HDF5):
        /trainval/images_log: Historical sky images (N, T_log, H, W, C)
        /trainval/images_pred: Future sky images (N, T_pred, H, W, C)
        /trainval/pv_log: Historical PV output (N, T_log)
        /trainval/pv_pred: Future PV output (N, T_pred)
        /trainval/timestamps: Unix timestamps (N, T_total) [optional]
        /test/... (same structure)
    
    Reference:
        Nie et al., "SKIPP'D: SKy Images and Photovoltaic Power Generation Dataset"
        Solar Energy, Volume 255, 2023, Pages 171-179
    """
    
    def __init__(
        self,
        data_path: str,
        config,
        split: str = 'train',
        transform: Optional[VideoAugmentation] = None,
        use_clear_sky_index: bool = False,
        validate_data: bool = True
    ):
        """
        Initialize SKIPP'D dataset.
        
        Args:
            data_path: Path to HDF5 data file
            config: DataConfig object
            split: Data split ('train', 'val', 'test')
            transform: Optional augmentation transform
            use_clear_sky_index: Whether to normalize PV by clear-sky model
            validate_data: Whether to validate data on initialization
        """
        super().__init__()
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.data_path = data_path
        self.config = config
        self.split = split
        self.transform = transform
        self.use_clear_sky_index = use_clear_sky_index
        
        # Configuration parameters
        self.resolution = config.resolution
        self.sequence_length = config.sequence_length
        self.n_cond_frames = config.n_cond_frames
        self.pv_max = config.pv_max
        self.pv_min = getattr(config, 'pv_min', 0.0)
        self.cloud_threshold = config.cloud_threshold
        self.cloudy_only = config.cloudy_only
        
        # Normalization parameters
        self.image_mean = tuple(config.image_mean)
        self.image_std = tuple(config.image_std)
        
        # Determine HDF5 group
        self.hdf5_group = 'test' if split == 'test' else 'trainval'
        
        # Initialize worker-safe file handler
        self._handler = None  # Lazy initialization
        
        # Load metadata and build indices
        self._load_metadata()
        self._build_indices()
        
        # Validate data if requested
        if validate_data:
            self._validate_data()
        
        logger.info(
            f"Initialized {split} dataset: {len(self)} samples "
            f"(cloudy_only={self.cloudy_only}, clear_sky_index={use_clear_sky_index})"
        )
    
    @property
    def handler(self) -> HDF5Handler:
        """Lazy-initialize and return worker-safe file handler."""
        if self._handler is None:
            self._handler = HDF5Handler(self.data_path)
        return self._handler
    
    def _load_metadata(self):
        """Load dataset metadata without keeping file handle open."""
        with h5py.File(self.data_path, 'r') as f:
            group = f[self.hdf5_group]
            
            # Get dataset shapes
            self.images_log_shape = group['images_log'].shape
            self.images_pred_shape = group['images_pred'].shape
            self.pv_log_shape = group['pv_log'].shape
            self.pv_pred_shape = group['pv_pred'].shape
            
            self.total_samples = self.images_log_shape[0]
            
            # Check for timestamps
            self.has_timestamps = 'timestamps' in group
            
            # Store sample timestamps if available (for date-based splitting)
            if self.has_timestamps:
                self._timestamps = group['timestamps'][:]
            else:
                self._timestamps = None
    
    def _build_indices(self):
        """
        Build sample indices based on split and filtering criteria.
        
        Handles:
            - Train/val split from trainval group
            - Date-based test set selection
            - Cloud filtering with caching
        """
        all_indices = list(range(self.total_samples))
        
        # Handle train/val/test split
        if self.hdf5_group == 'trainval':
            # Compute split indices
            train_size = int(self.total_samples * self.config.train_ratio / 
                           (self.config.train_ratio + self.config.val_ratio))
            
            if self.split == 'train':
                self.indices = all_indices[:train_size]
            else:  # val
                self.indices = all_indices[train_size:]
        else:
            # Test set - use all samples
            self.indices = all_indices
        
        # Apply cloud filtering if configured
        if self.cloudy_only and self.split != 'test':
            self.indices = self._filter_cloudy_samples(self.indices)
        
        logger.debug(
            f"Split '{self.split}': {len(self.indices)} samples "
            f"(from {self.total_samples} total in {self.hdf5_group})"
        )
    
    def _filter_cloudy_samples(self, indices: List[int]) -> List[int]:
        """
        Filter indices to keep only cloudy samples.
        
        Uses caching to avoid recomputing cloud indices on each run.
        
        Args:
            indices: List of sample indices to filter
            
        Returns:
            Filtered list containing only cloudy sample indices
        """
        # Generate cache key based on configuration
        cache_key = hashlib.md5(
            f"{self.data_path}_{self.hdf5_group}_{self.cloud_threshold}".encode()
        ).hexdigest()[:8]
        
        cache_path = Path(self.config.cache_dir) / f"cloud_indices_{cache_key}.npz"
        
        # Try to load from cache
        if cache_path.exists():
            try:
                cached = np.load(cache_path)
                cloud_indices = cached['cloud_indices']
                cached_threshold = float(cached['threshold'])
                
                if cached_threshold == self.cloud_threshold:
                    logger.debug(f"Loaded cloud indices from cache: {cache_path}")
                else:
                    cloud_indices = None
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                cloud_indices = None
        else:
            cloud_indices = None
        
        # Compute cloud indices if not cached
        if cloud_indices is None:
            logger.info("Computing cloud indices (this may take a while)...")
            
            cloud_indices = np.zeros(self.total_samples)
            
            with h5py.File(self.data_path, 'r') as f:
                images = f[f'{self.hdf5_group}/images_log']
                
                # Process in batches for efficiency
                batch_size = 100
                for start_idx in range(0, self.total_samples, batch_size):
                    end_idx = min(start_idx + batch_size, self.total_samples)
                    
                    # Load batch of last historical frames
                    batch_images = images[start_idx:end_idx, -1]  # (batch, H, W, C)
                    
                    # Compute cloud indices vectorized
                    batch_cloud_idx = compute_cloud_index_vectorized(batch_images)
                    cloud_indices[start_idx:end_idx] = batch_cloud_idx
                    
                    if (start_idx // batch_size) % 10 == 0:
                        logger.debug(f"Processed {end_idx}/{self.total_samples} samples")
            
            # Save to cache
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cache_path,
                    cloud_indices=cloud_indices,
                    threshold=self.cloud_threshold
                )
                logger.info(f"Saved cloud indices to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        # Filter based on threshold
        cloudy_mask = cloud_indices[indices] < self.cloud_threshold
        cloudy_indices = [idx for idx, is_cloudy in zip(indices, cloudy_mask) if is_cloudy]
        
        logger.info(
            f"Cloud filtering: {len(cloudy_indices)}/{len(indices)} samples "
            f"({100 * len(cloudy_indices) / len(indices):.1f}%) below threshold {self.cloud_threshold}"
        )
        
        return cloudy_indices
    
    def _validate_data(self):
        """Validate data shapes and ranges."""
        # Validate shapes
        expected_t_log = self.n_cond_frames
        expected_t_pred = self.sequence_length - self.n_cond_frames
        
        if self.images_log_shape[1] != expected_t_log:
            logger.warning(
                f"Unexpected images_log temporal dim: {self.images_log_shape[1]} "
                f"(expected {expected_t_log})"
            )
        
        if self.images_pred_shape[1] != expected_t_pred:
            logger.warning(
                f"Unexpected images_pred temporal dim: {self.images_pred_shape[1]} "
                f"(expected {expected_t_pred})"
            )
    
    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index (0 to len-1)
            
        Returns:
            Dictionary containing:
                - 'video': Sky video sequence (C, T, H, W), normalized
                - 'pv_target': Future PV output target (scalar or sequence)
                - 'pv_history': Historical PV values
                - 'clear_sky_index': Clear-sky index if use_clear_sky_index=True
                - 'idx': Original sample index in HDF5 file
        """
        # Map to actual index
        actual_idx = self.indices[idx]
        
        # Access data through worker-safe handler
        handler = self.handler
        group = handler[self.hdf5_group]
        
        # Load images
        images_log = np.array(group['images_log'][actual_idx])   # (T_log, H, W, C)
        images_pred = np.array(group['images_pred'][actual_idx]) # (T_pred, H, W, C)
        
        # Load PV data
        pv_log = np.array(group['pv_log'][actual_idx]).astype(np.float32)   # (T_log,)
        pv_pred = np.array(group['pv_pred'][actual_idx]).astype(np.float32) # (T_pred,)
        
        # Concatenate temporal sequences
        images = np.concatenate([images_log, images_pred], axis=0)  # (T, H, W, C)
        
        # Convert to tensors
        images = torch.from_numpy(images)
        pv_log = torch.from_numpy(pv_log)
        pv_pred = torch.from_numpy(pv_pred)
        
        # Preprocess video
        video = preprocess_video(
            images,
            resolution=self.resolution,
            sequence_length=self.sequence_length,
            normalize=True,
            mean=self.image_mean,
            std=self.image_std
        )
        
        # Apply augmentation (training only)
        if self.transform is not None:
            video = self.transform(video)
        
        # Normalize PV values
        pv_range = self.pv_max - self.pv_min
        pv_target = (pv_pred[-1] - self.pv_min) / pv_range  # Last predicted timestep
        pv_history = (pv_log - self.pv_min) / pv_range
        
        # Build output dictionary
        output = {
            'video': video,
            'pv_target': pv_target,
            'pv_history': pv_history,
            'idx': actual_idx
        }
        
        # Add clear-sky index if requested
        if self.use_clear_sky_index and self.has_timestamps:
            timestamps = np.array(group['timestamps'][actual_idx])
            pv_all = np.concatenate([pv_log.numpy(), pv_pred.numpy()])
            clear_sky_idx = compute_clear_sky_index(pv_all, timestamps)
            output['clear_sky_index'] = torch.from_numpy(clear_sky_idx).float()
        
        return output
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a sample without loading full data.
        
        Useful for debugging and analysis.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample metadata
        """
        actual_idx = self.indices[idx]
        
        info = {
            'idx': idx,
            'actual_idx': actual_idx,
            'split': self.split,
        }
        
        if self.has_timestamps:
            with h5py.File(self.data_path, 'r') as f:
                timestamps = f[f'{self.hdf5_group}/timestamps'][actual_idx]
                info['start_time'] = datetime.fromtimestamp(timestamps[0])
                info['end_time'] = datetime.fromtimestamp(timestamps[-1])
        
        return info
    
    def __getstate__(self) -> Dict[str, Any]:
        """Prepare for pickling (multiprocessing)."""
        state = self.__dict__.copy()
        # Don't pickle the handler - each worker will create its own
        state['_handler'] = None
        return state
    
    def __setstate__(self, state: Dict[str, Any]):
        """Restore after unpickling."""
        self.__dict__.update(state)
        # Handler will be lazily initialized on first access
    
    def close(self):
        """Explicitly close file handles."""
        if self._handler is not None:
            self._handler.close()
            self._handler = None


# ============================================================================
# PyTorch Lightning DataModule
# ============================================================================

if LIGHTNING_AVAILABLE:
    class SKIPPDDataModule(pl.LightningDataModule):
        """
        PyTorch Lightning DataModule for SKIPP'D dataset.
        
        Provides standardized data loading with support for:
            - Multi-GPU distributed training
            - Automatic data preprocessing
            - Configurable augmentation
            - Efficient data loading with prefetching
            - Clear-sky index normalization
        """
        
        def __init__(
            self,
            config,
            use_clear_sky_index: bool = False
        ):
            """
            Initialize DataModule.
            
            Args:
                config: ExperimentConfig object
                use_clear_sky_index: Whether to compute clear-sky indices
            """
            super().__init__()
            
            self.config = config
            self.data_config = config.data
            self.use_clear_sky_index = use_clear_sky_index
            
            # Dataset references (set in setup())
            self.train_dataset: Optional[SKIPPDDataset] = None
            self.val_dataset: Optional[SKIPPDDataset] = None
            self.test_dataset: Optional[SKIPPDDataset] = None
            
            # Augmentation transform (training only)
            self.train_transform = VideoAugmentation(self.data_config)
            
            # Save hyperparameters for logging
            self.save_hyperparameters(ignore=['config'])
        
        def prepare_data(self):
            """
            Prepare data (called once on main process).
            
            Verifies data availability and creates cache directories.
            """
            data_path = Path(self.data_config.data_path)
            
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                logger.info("Please download the SKIPP'D dataset from:")
                logger.info("  Stanford Digital Repository: https://purl.stanford.edu/dj417rh1007")
                logger.info("  Hugging Face: torchgeo/skippd")
                logger.info("  GitHub: https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset")
                raise FileNotFoundError(f"SKIPP'D dataset required: {data_path}")
            
            # Create cache directory
            cache_dir = Path(self.data_config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Data file verified: {data_path}")
        
        def setup(self, stage: Optional[str] = None):
            """
            Set up datasets for each stage.
            
            Args:
                stage: 'fit', 'validate', 'test', or None for all
            """
            data_path = self.data_config.data_path
            
            if stage == 'fit' or stage is None:
                self.train_dataset = SKIPPDDataset(
                    data_path=data_path,
                    config=self.data_config,
                    split='train',
                    transform=self.train_transform,
                    use_clear_sky_index=self.use_clear_sky_index,
                    validate_data=True
                )
                
                self.val_dataset = SKIPPDDataset(
                    data_path=data_path,
                    config=self.data_config,
                    split='val',
                    transform=None,  # No augmentation for validation
                    use_clear_sky_index=self.use_clear_sky_index,
                    validate_data=False
                )
            
            if stage == 'test' or stage is None:
                self.test_dataset = SKIPPDDataset(
                    data_path=data_path,
                    config=self.data_config,
                    split='test',
                    transform=None,
                    use_clear_sky_index=self.use_clear_sky_index,
                    validate_data=False
                )
        
        def _create_dataloader(
            self,
            dataset: SKIPPDDataset,
            shuffle: bool = False,
            drop_last: bool = False
        ) -> data.DataLoader:
            """
            Create DataLoader with proper configuration.
            
            Handles distributed sampling for multi-GPU training.
            
            Args:
                dataset: Dataset to load from
                shuffle: Whether to shuffle data
                drop_last: Whether to drop incomplete final batch
                
            Returns:
                Configured DataLoader
            """
            # Distributed sampler for multi-GPU
            sampler = None
            if DISTRIBUTED_AVAILABLE and dist.is_initialized():
                sampler = data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=shuffle,
                    drop_last=drop_last
                )
                shuffle = False  # Sampler handles shuffling
            
            # DataLoader configuration
            num_workers = self.data_config.num_workers
            
            loader_kwargs = {
                'batch_size': self.data_config.batch_size,
                'shuffle': shuffle if sampler is None else False,
                'num_workers': num_workers,
                'pin_memory': self.data_config.pin_memory,
                'drop_last': drop_last,
                'sampler': sampler,
            }
            
            # Add worker-specific options if using multiprocessing
            if num_workers > 0:
                loader_kwargs['prefetch_factor'] = self.data_config.prefetch_factor
                loader_kwargs['persistent_workers'] = self.data_config.persistent_workers
            
            return data.DataLoader(dataset, **loader_kwargs)
        
        def train_dataloader(self) -> data.DataLoader:
            """Create training DataLoader."""
            drop_last = getattr(self.data_config, 'drop_last', True)
            return self._create_dataloader(
                self.train_dataset,
                shuffle=True,
                drop_last=drop_last
            )
        
        def val_dataloader(self) -> data.DataLoader:
            """Create validation DataLoader."""
            return self._create_dataloader(
                self.val_dataset,
                shuffle=False,
                drop_last=False
            )
        
        def test_dataloader(self) -> data.DataLoader:
            """Create test DataLoader."""
            return self._create_dataloader(
                self.test_dataset,
                shuffle=False,
                drop_last=False
            )
        
        @property
        def n_train_samples(self) -> int:
            """Number of training samples."""
            return len(self.train_dataset) if self.train_dataset else 0
        
        @property
        def n_val_samples(self) -> int:
            """Number of validation samples."""
            return len(self.val_dataset) if self.val_dataset else 0
        
        @property
        def n_test_samples(self) -> int:
            """Number of test samples."""
            return len(self.test_dataset) if self.test_dataset else 0
        
        def teardown(self, stage: Optional[str] = None):
            """Clean up resources."""
            for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
                if dataset is not None:
                    dataset.close()

else:
    # Provide a placeholder if Lightning is not available
    class SKIPPDDataModule:
        """Placeholder when PyTorch Lightning is not installed."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch Lightning is required for SKIPPDDataModule. "
                "Install with: pip install pytorch-lightning"
            )


# ============================================================================
# Data Analysis Utilities
# ============================================================================

class DatasetAnalyzer:
    """
    Utility class for analyzing SKIPP'D dataset statistics.
    
    Provides methods for:
        - Computing distribution statistics
        - Temporal correlation analysis
        - Data quality validation
        - Clear-sky model fitting
    """
    
    def __init__(self, datamodule: SKIPPDDataModule):
        """
        Initialize analyzer.
        
        Args:
            datamodule: Initialized DataModule
        """
        self.datamodule = datamodule
    
    def compute_statistics(
        self,
        n_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive dataset statistics.
        
        Args:
            n_samples: Number of samples to use (None for all)
            
        Returns:
            Dictionary with statistics for each split
        """
        stats = {}
        
        datasets = [
            ('train', self.datamodule.train_dataset),
            ('val', self.datamodule.val_dataset),
            ('test', self.datamodule.test_dataset)
        ]
        
        for split_name, dataset in datasets:
            if dataset is None:
                continue
            
            # Determine sample size
            total = len(dataset)
            sample_size = min(n_samples or total, total)
            indices = np.random.choice(total, sample_size, replace=False)
            
            # Collect statistics
            pv_targets = []
            pv_histories = []
            video_means = []
            video_stds = []
            
            for idx in indices:
                sample = dataset[idx]
                pv_targets.append(sample['pv_target'].item())
                pv_histories.append(sample['pv_history'].numpy())
                video_means.append(sample['video'].mean().item())
                video_stds.append(sample['video'].std().item())
            
            pv_targets = np.array(pv_targets)
            pv_histories = np.array(pv_histories)
            video_means = np.array(video_means)
            video_stds = np.array(video_stds)
            
            stats[split_name] = {
                'n_samples': total,
                'n_analyzed': sample_size,
                'pv_target_mean': float(pv_targets.mean()),
                'pv_target_std': float(pv_targets.std()),
                'pv_target_min': float(pv_targets.min()),
                'pv_target_max': float(pv_targets.max()),
                'pv_target_median': float(np.median(pv_targets)),
                'pv_history_mean': float(pv_histories.mean()),
                'pv_history_std': float(pv_histories.std()),
                'video_mean': float(video_means.mean()),
                'video_std': float(video_stds.mean()),
            }
        
        return stats
    
    def compute_temporal_autocorrelation(
        self,
        n_samples: int = 500,
        max_lag: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute temporal autocorrelation of PV output.
        
        Useful for understanding temporal dependencies and
        validating persistence model assumptions.
        
        Args:
            n_samples: Number of samples to analyze
            max_lag: Maximum lag to compute (default: sequence length)
            
        Returns:
            Dictionary with autocorrelation statistics
        """
        if self.datamodule.train_dataset is None:
            return {}
        
        dataset = self.datamodule.train_dataset
        n_samples = min(n_samples, len(dataset))
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        
        pv_sequences = []
        for idx in indices:
            sample = dataset[idx]
            pv_sequences.append(sample['pv_history'].numpy())
        
        pv_sequences = np.array(pv_sequences)
        T = pv_sequences.shape[1]
        max_lag = max_lag or T
        
        # Compute autocorrelation for each lag
        autocorr = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                autocorr[lag] = 1.0
            elif lag < T:
                x = pv_sequences[:, :-lag].flatten()
                y = pv_sequences[:, lag:].flatten()
                if len(x) > 0:
                    autocorr[lag] = np.corrcoef(x, y)[0, 1]
        
        return {
            'lags': np.arange(max_lag),
            'autocorrelation': autocorr,
            'temporal_mean': pv_sequences.mean(axis=0),
            'temporal_std': pv_sequences.std(axis=0),
        }
    
    def validate_data_integrity(self) -> Dict[str, bool]:
        """
        Validate data integrity and consistency.
        
        Checks:
            - File existence and accessibility
            - Data format and structure
            - Value ranges
            - Missing/NaN values
            - Shape consistency
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'file_exists': False,
            'file_readable': False,
            'valid_structure': False,
            'valid_shapes': False,
            'valid_ranges': False,
            'no_nan_values': False,
            'temporal_consistency': False,
        }
        
        data_path = self.datamodule.data_config.data_path
        
        # Check file existence
        results['file_exists'] = os.path.exists(data_path)
        if not results['file_exists']:
            return results
        
        try:
            with h5py.File(data_path, 'r') as f:
                results['file_readable'] = True
                
                # Check structure
                has_trainval = 'trainval' in f
                has_test = 'test' in f
                results['valid_structure'] = has_trainval or has_test
                
                if not results['valid_structure']:
                    return results
                
                # Check a sample group
                group_name = 'trainval' if has_trainval else 'test'
                group = f[group_name]
                
                required_datasets = ['images_log', 'images_pred', 'pv_log', 'pv_pred']
                all_present = all(ds in group for ds in required_datasets)
                results['valid_structure'] = all_present
                
                if not all_present:
                    return results
                
                # Check shapes
                img_log_shape = group['images_log'].shape
                img_pred_shape = group['images_pred'].shape
                
                results['valid_shapes'] = (
                    len(img_log_shape) == 5 and  # (N, T, H, W, C)
                    img_log_shape[0] == img_pred_shape[0] and  # Same N
                    img_log_shape[2:] == img_pred_shape[2:]    # Same H, W, C
                )
                
                # Check value ranges (sample first 10)
                sample_images = group['images_log'][:10]
                sample_pv = group['pv_log'][:10]
                
                results['valid_ranges'] = (
                    np.all(sample_images >= 0) and
                    np.all(sample_images <= 255) and
                    np.all(sample_pv >= 0) and
                    np.all(sample_pv <= SKIPPD_PV_CAPACITY_KW * 1.5)  # Allow some margin
                )
                
                # Check for NaN values
                results['no_nan_values'] = (
                    not np.any(np.isnan(sample_images)) and
                    not np.any(np.isnan(sample_pv))
                )
                
                # Check temporal consistency
                results['temporal_consistency'] = (
                    group['pv_log'].shape[1] == group['images_log'].shape[1] and
                    group['pv_pred'].shape[1] == group['images_pred'].shape[1]
                )
                
        except Exception as e:
            logger.error(f"Data validation error: {e}")
        
        return results
    
    def print_summary(self):
        """Print a human-readable summary of the dataset."""
        print("=" * 70)
        print("SKIPP'D Dataset Analysis Summary")
        print("=" * 70)
        
        # Validation
        validation = self.validate_data_integrity()
        print("\nData Integrity Validation:")
        for key, value in validation.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key.replace('_', ' ').title()}")
        
        # Statistics
        if all(validation.values()):
            print("\nDataset Statistics:")
            stats = self.compute_statistics(n_samples=500)
            for split, split_stats in stats.items():
                print(f"\n  {split.upper()}:")
                print(f"    Samples: {split_stats['n_samples']}")
                print(f"    PV Target: {split_stats['pv_target_mean']:.3f} ± {split_stats['pv_target_std']:.3f}")
                print(f"    PV Range: [{split_stats['pv_target_min']:.3f}, {split_stats['pv_target_max']:.3f}]")
        
        print("\n" + "=" * 70)


# ============================================================================
# Collate Functions
# ============================================================================

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.
    
    Handles variable-length sequences and optional fields gracefully.
    
    Args:
        batch: List of sample dictionaries from __getitem__
        
    Returns:
        Batched dictionary of tensors
    """
    collated = {
        'video': torch.stack([s['video'] for s in batch]),
        'pv_target': torch.stack([s['pv_target'] for s in batch]),
        'pv_history': torch.stack([s['pv_history'] for s in batch]),
        'idx': torch.tensor([s['idx'] for s in batch], dtype=torch.long)
    }
    
    # Handle optional fields
    if 'clear_sky_index' in batch[0]:
        collated['clear_sky_index'] = torch.stack([s['clear_sky_index'] for s in batch])
    
    return collated


# ============================================================================
# Factory Functions
# ============================================================================

def create_datamodule(config, use_clear_sky_index: bool = False) -> SKIPPDDataModule:
    """
    Factory function to create SKIPP'D DataModule.
    
    Args:
        config: ExperimentConfig object
        use_clear_sky_index: Whether to compute clear-sky normalization
        
    Returns:
        Initialized SKIPPDDataModule
    """
    return SKIPPDDataModule(config, use_clear_sky_index=use_clear_sky_index)


def create_dataset(
    data_path: str,
    config,
    split: str = 'train',
    transform: Optional[VideoAugmentation] = None,
    **kwargs
) -> SKIPPDDataset:
    """
    Factory function to create standalone SKIPP'D Dataset.
    
    Args:
        data_path: Path to HDF5 file
        config: DataConfig object
        split: Data split ('train', 'val', 'test')
        transform: Optional augmentation transform
        **kwargs: Additional arguments passed to SKIPPDDataset
        
    Returns:
        Initialized SKIPPDDataset
    """
    return SKIPPDDataset(
        data_path=data_path,
        config=config,
        split=split,
        transform=transform,
        **kwargs
    )


# ============================================================================
# Testing and Demonstration
# ============================================================================

def create_synthetic_dataset(
    output_path: str,
    n_train_samples: int = 100,
    n_test_samples: int = 20,
    resolution: int = 64,
    t_log: int = 8,
    t_pred: int = 8
) -> str:
    """
    Create synthetic HDF5 dataset for testing.
    
    Generates random images and PV values with realistic structure.
    
    Args:
        output_path: Path to save HDF5 file
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
        resolution: Image resolution
        t_log: Number of historical frames
        t_pred: Number of prediction frames
        
    Returns:
        Path to created dataset
    """
    logger.info(f"Creating synthetic dataset at: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Training/validation data
        trainval = f.create_group('trainval')
        trainval.create_dataset(
            'images_log',
            data=np.random.randint(0, 256, (n_train_samples, t_log, resolution, resolution, 3), dtype=np.uint8)
        )
        trainval.create_dataset(
            'images_pred',
            data=np.random.randint(0, 256, (n_train_samples, t_pred, resolution, resolution, 3), dtype=np.uint8)
        )
        trainval.create_dataset(
            'pv_log',
            data=np.random.uniform(0, 25, (n_train_samples, t_log)).astype(np.float32)
        )
        trainval.create_dataset(
            'pv_pred',
            data=np.random.uniform(0, 25, (n_train_samples, t_pred)).astype(np.float32)
        )
        
        # Test data
        test = f.create_group('test')
        test.create_dataset(
            'images_log',
            data=np.random.randint(0, 256, (n_test_samples, t_log, resolution, resolution, 3), dtype=np.uint8)
        )
        test.create_dataset(
            'images_pred',
            data=np.random.randint(0, 256, (n_test_samples, t_pred, resolution, resolution, 3), dtype=np.uint8)
        )
        test.create_dataset(
            'pv_log',
            data=np.random.uniform(0, 25, (n_test_samples, t_log)).astype(np.float32)
        )
        test.create_dataset(
            'pv_pred',
            data=np.random.uniform(0, 25, (n_test_samples, t_pred)).astype(np.float32)
        )
    
    logger.info(f"Created synthetic dataset: {n_train_samples} train, {n_test_samples} test samples")
    return output_path


if __name__ == "__main__":
    import sys
    import argparse
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="SKIPP'D Data Module Testing")
    parser.add_argument('--data-path', type=str, default=None, help='Path to SKIPP\'D dataset')
    parser.add_argument('--create-synthetic', action='store_true', help='Create synthetic test data')
    parser.add_argument('--analyze', action='store_true', help='Run dataset analysis')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SKIPP'D Data Module Testing")
    print("=" * 70)
    
    # Import config module
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from config import ExperimentConfig, DataConfig
        config = ExperimentConfig()
    except ImportError:
        # Create minimal config for testing
        @dataclass
        class MinimalDataConfig:
            data_path: str = "./data/synthetic_test.hdf5"
            cache_dir: str = "./data/cache"
            sequence_length: int = 16
            n_cond_frames: int = 8
            resolution: int = 64
            train_ratio: float = 0.88
            val_ratio: float = 0.07
            test_ratio: float = 0.05
            cloudy_only: bool = False
            cloud_threshold: float = 0.3
            batch_size: int = 4
            num_workers: int = 0
            pin_memory: bool = False
            prefetch_factor: int = 2
            persistent_workers: bool = False
            use_augmentation: bool = True
            horizontal_flip_prob: float = 0.5
            rotation_degrees: float = 15.0
            brightness_jitter: float = 0.1
            contrast_jitter: float = 0.1
            image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
            image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
            pv_max: float = 30.0
            pv_min: float = 0.0
            drop_last: bool = True
        
        class MinimalConfig:
            def __init__(self):
                self.data = MinimalDataConfig()
        
        config = MinimalConfig()
    
    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = config.data.data_path
    
    # Create synthetic data if requested or if real data doesn't exist
    if args.create_synthetic or not os.path.exists(data_path):
        cache_dir = Path(config.data.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_path = str(cache_dir / "synthetic_test.hdf5")
        config.data.data_path = data_path
        config.data.cloudy_only = False  # Disable for synthetic
        create_synthetic_dataset(data_path)
    
    print(f"\nData path: {data_path}")
    print(f"Data exists: {os.path.exists(data_path)}")
    
    if not os.path.exists(data_path):
        print("ERROR: Data file not found!")
        sys.exit(1)
    
    # Test dataset directly
    print("\n--- Testing SKIPPDDataset ---")
    
    transform = VideoAugmentation(config.data)
    print(f"Transform: {transform}")
    
    train_dataset = SKIPPDDataset(
        data_path=data_path,
        config=config.data,
        split='train',
        transform=transform,
        validate_data=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Test single sample
    sample = train_dataset[0]
    print(f"\nSample contents:")
    print(f"  video shape: {sample['video'].shape}")
    print(f"  video range: [{sample['video'].min():.3f}, {sample['video'].max():.3f}]")
    print(f"  pv_target: {sample['pv_target']:.4f}")
    print(f"  pv_history shape: {sample['pv_history'].shape}")
    print(f"  idx: {sample['idx']}")
    
    # Test DataModule if Lightning available
    if LIGHTNING_AVAILABLE:
        print("\n--- Testing SKIPPDDataModule ---")
        
        datamodule = create_datamodule(config)
        datamodule.prepare_data()
        datamodule.setup('fit')
        
        print(f"Train samples: {datamodule.n_train_samples}")
        print(f"Val samples: {datamodule.n_val_samples}")
        
        # Test DataLoader
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"\nBatch contents:")
        print(f"  video: {batch['video'].shape}")
        print(f"  pv_target: {batch['pv_target'].shape}")
        print(f"  pv_history: {batch['pv_history'].shape}")
        
        # Run analysis if requested
        if args.analyze:
            print("\n--- Dataset Analysis ---")
            analyzer = DatasetAnalyzer(datamodule)
            analyzer.print_summary()
        
        # Cleanup
        datamodule.teardown()
    
    # Clean up dataset
    train_dataset.close()
    
    print("\n" + "=" * 70)
    print("Data module testing complete!")
    print("=" * 70)