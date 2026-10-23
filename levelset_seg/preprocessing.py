"""Image preprocessing: scaling, CLAHE, Gaussian blur, and K-Means clustering.

All operations match the original ``main.py`` pipeline exactly.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import SolverConfig


def scale_image(
    image: NDArray[np.uint8],
    max_dim: int,
) -> NDArray[np.uint8]:
    """Down-scale *image* so that ``max(h, w) <= max_dim``.

    Uses area-based interpolation (``cv2.INTER_AREA``).  If the image
    already fits, it is returned unchanged (no copy).

    Parameters
    ----------
    image : ndarray of uint8, shape (H, W)
        Grayscale input image.
    max_dim : int
        Maximum allowed dimension.

    Returns
    -------
    ndarray of uint8, shape (H', W')
    """
    h, w = image.shape
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    return cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def enhance_contrast(
    image: NDArray[np.uint8],
    clip_limit: float,
    grid_size: Tuple[int, int],
    blur_ksize: Tuple[int, int],
) -> NDArray[np.uint8]:
    """Apply CLAHE contrast enhancement followed by Gaussian blur.

    Parameters
    ----------
    image : ndarray of uint8, shape (H, W)
        Grayscale input image.
    clip_limit : float
        CLAHE clip-limit parameter.
    grid_size : (int, int)
        CLAHE tile grid size.
    blur_ksize : (int, int)
        Gaussian blur kernel size.

    Returns
    -------
    ndarray of uint8, shape (H, W)
        Enhanced and smoothed image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    img_clahe = clahe.apply(image)
    return cv2.GaussianBlur(img_clahe, blur_ksize, 0)


def kmeans_cluster(
    image: NDArray[np.uint8],
    k: int,
    max_iter: int,
    epsilon: float,
    attempts: int,
) -> Tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.uint8]]:
    """Segment pixel intensities into *k* clusters via K-Means.

    Parameters
    ----------
    image : ndarray of uint8, shape (H, W)
        Grayscale input (typically after CLAHE + blur).
    k : int
        Number of clusters.
    max_iter : int
        Maximum iterations per K-Means attempt.
    epsilon : float
        Convergence threshold.
    attempts : int
        Number of K-Means restarts.

    Returns
    -------
    labels : ndarray of int32, shape (H*W, 1)
        Cluster label for every pixel.
    centers : ndarray of float32, shape (k, 1)
        Cluster centres (intensity values).
    vis : ndarray of uint8, shape (H, W)
        Visualisation image (labels * 51, clipped to uint8).
    """
    pixel_values = np.float32(image.reshape((-1, 1)))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        epsilon,
    )
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS
    )
    vis = (labels.flatten() * 51).reshape(image.shape).astype(np.uint8)
    return labels, centers, vis


def preprocess(
    image_path: str,
    cfg: SolverConfig | None = None,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.uint8],
    NDArray[np.uint8],
]:
    """Full preprocessing pipeline: load → scale → CLAHE → K-Means.

    Parameters
    ----------
    image_path : str
        Path to the input MRI image.
    cfg : SolverConfig, optional
        Configuration object.  Uses defaults if *None*.

    Returns
    -------
    image : ndarray of float64, shape (H, W)
        Normalised [0, 1] image fed to the PDE solver.
    labels : ndarray of int32, shape (H*W, 1)
        K-Means cluster labels.
    centers : ndarray of float32, shape (K, 1)
        K-Means cluster centres.
    kmeans_vis : ndarray of uint8, shape (H, W)
        K-Means visualisation.
    img_raw_scaled : ndarray of uint8, shape (H, W)
        Grayscale image after scaling (before CLAHE), kept for
        mask operations that reference the raw spatial size.
    """
    if cfg is None:
        cfg = SolverConfig()

    img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_raw = scale_image(img_raw, cfg.max_dim)
    img_smooth = enhance_contrast(
        img_raw, cfg.clahe_clip_limit, cfg.clahe_grid_size, cfg.blur_ksize
    )
    image = img_smooth.astype(np.float64) / 255.0
    labels, centers, kmeans_vis = kmeans_cluster(
        img_smooth,
        cfg.kmeans_k,
        cfg.kmeans_max_iter,
        cfg.kmeans_epsilon,
        cfg.kmeans_attempts,
    )
    return image, labels, centers, kmeans_vis, img_raw
