"""Default parameters and configuration for the segmentation pipeline.

Every constant here is verified against the values used in the original
single-file ``main.py``.  Changing a default changes the segmentation
output — do so deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SolverConfig:
    """All tuneable parameters for the segmentation pipeline.

    Attributes
    ----------
    total_iterations : int
        Number of PDE evolution steps (default 85).
    dt : float
        Euler-method time step for the level-set update (default 0.5).
    lambda1 : float
        Weight for the inside-region data-fidelity term (default 1.0).
    lambda2 : float
        Weight for the outside-region data-fidelity term (default 1.0).
    epsilon : float
        Regularisation width for the smoothed Heaviside and Dirac
        functions (default 1.0).
    phi_clip : float
        Symmetric clamp applied to φ after every iteration and at
        initialisation to keep the SDF bounded (default 5.0).
    max_dim : int
        Images are downscaled so that ``max(height, width) <= max_dim``
        before solving (default 400).
    kmeans_k : int
        Number of K-Means clusters (default 5).
    kmeans_attempts : int
        Number of K-Means restarts passed to ``cv2.kmeans`` (default 10).
    kmeans_max_iter : int
        Maximum iterations per K-Means run (default 100).
    kmeans_epsilon : float
        K-Means convergence threshold (default 0.2).
    clahe_clip_limit : float
        CLAHE contrast-limit parameter (default 2.0).
    clahe_grid_size : Tuple[int, int]
        CLAHE tile grid size (default (8, 8)).
    blur_ksize : Tuple[int, int]
        Gaussian blur kernel size (default (5, 5)).
    morph_ksize : Tuple[int, int]
        Structuring-element size for morphological opening of cluster
        masks during ROI search (default (5, 5)).
    min_contour_area : float
        Minimum contour area (pixels) to consider during ROI search
        (default 100).
    top_n_clusters : int
        Number of brightest K-Means clusters to search for candidate
        ROIs (default 3).
    convergence_tol : float or None
        If not *None*, the solver stops early when
        ``max|φ_new − φ_old| < convergence_tol``.  *None* (default)
        means the solver always runs for exactly *total_iterations*.
    variance_threshold : float
        Image-variance threshold for the adaptive curvature weight μ.
        If ``np.var(image) < variance_threshold`` then μ = 0.5,
        otherwise μ = 0.25  (default 0.05).
    mu_low_variance : float
        Curvature weight used when image variance is below the
        threshold (default 0.5).
    mu_high_variance : float
        Curvature weight used when image variance is at or above the
        threshold (default 0.25).
    gui_poll_ms : int
        GUI polling interval in milliseconds (default 50).
    gui_redraw_every : int
        Redraw the Matplotlib canvas every N solver iterations
        (default 10).
    """

    # PDE solver
    total_iterations: int = 85
    dt: float = 0.5
    lambda1: float = 1.0
    lambda2: float = 1.0
    epsilon: float = 1.0
    phi_clip: float = 5.0

    # Image preprocessing
    max_dim: int = 400
    kmeans_k: int = 5
    kmeans_attempts: int = 10
    kmeans_max_iter: int = 100
    kmeans_epsilon: float = 0.2
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    blur_ksize: Tuple[int, int] = (5, 5)

    # ROI search
    morph_ksize: Tuple[int, int] = (5, 5)
    min_contour_area: float = 100
    top_n_clusters: int = 3

    # Convergence stopping (off by default)
    convergence_tol: float | None = None

    # Adaptive curvature weight
    variance_threshold: float = 0.05
    mu_low_variance: float = 0.5
    mu_high_variance: float = 0.25

    # GUI
    gui_poll_ms: int = 50
    gui_redraw_every: int = 10
