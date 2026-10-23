"""Automatic ROI selection via contour scoring.

Scans the *top_n_clusters* brightest K-Means clusters for candidate
contours and picks the one that maximises

    score = √area · solidity³ · circularity³

where *solidity* = area / convex-hull area and *circularity* =
4π·area / perimeter².  This heuristic favours large, convex, roughly
circular regions — typical of meningioma-like tumors in MRI slices.

All operations match the original ``main.py`` pipeline exactly.

"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import SolverConfig


def find_best_roi(
    labels: NDArray[np.int32],
    centers: NDArray[np.float32],
    image_shape: Tuple[int, int],
    cfg: SolverConfig | None = None,
) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], float]:
    """Select the best tumor-candidate ROI from K-Means output.

    Parameters
    ----------
    labels : ndarray of int32, shape (H*W, 1)
        K-Means cluster labels.
    centers : ndarray of float32, shape (K, 1)
        K-Means cluster centres.
    image_shape : (int, int)
        Spatial shape ``(H, W)`` of the input image.
    cfg : SolverConfig, optional
        Configuration object.  Uses defaults if *None*.

    Returns
    -------
    mask_clean : ndarray of uint8, shape (H, W)
        Binary mask (0 / 255) of the selected ROI.
    topology_vis : ndarray of uint8, shape (H, W, 3)
        BGR visualisation with cyan-filled ROI and green convex hull.
    best_solidity : float
        Solidity of the selected contour (0 if no contour found).
    """
    if cfg is None:
        cfg = SolverConfig()

    sorted_centers_idx = np.argsort(centers.flatten())
    search_clusters = sorted_centers_idx[-cfg.top_n_clusters :]

    best_contour = None
    max_score: float = -1
    best_solidity: float = 0.0
    topology_vis = cv2.cvtColor(
        np.zeros(image_shape, dtype=np.uint8), cv2.COLOR_GRAY2RGB
    )
    mask_clean = np.zeros(image_shape, dtype=np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, cfg.morph_ksize
    )

    for cluster_idx in search_clusters:
        mask = (
            (labels.flatten() == cluster_idx)
            .reshape(image_shape)
            .astype(np.uint8)
            * 255
        )
        mask_opened = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        contours, _ = cv2.findContours(
            mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.min_contour_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter**2)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area

            score = np.sqrt(area) * (solidity**3) * (circularity**3)
            if score > max_score:
                max_score = score
                best_contour = cnt
                best_solidity = solidity

    if best_contour is not None:
        cv2.drawContours(
            mask_clean, [best_contour], -1, 255, thickness=cv2.FILLED
        )
        cv2.drawContours(
            topology_vis,
            [best_contour],
            -1,
            (0, 255, 255),
            thickness=cv2.FILLED,
        )
        cv2.drawContours(
            topology_vis,
            [cv2.convexHull(best_contour)],
            -1,
            (0, 255, 0),
            2,
        )
    else:
        # Fallback: use the brightest cluster as the entire mask
        mask_clean = (
            (labels.flatten() == sorted_centers_idx[-1])
            .reshape(image_shape)
            .astype(np.uint8)
            * 255
        )

    return mask_clean, topology_vis, best_solidity


def initialise_phi(
    mask: NDArray[np.uint8],
    phi_clip: float,
) -> NDArray[np.float64]:
    """Build a clipped signed distance function from a binary mask.

    Parameters
    ----------
    mask : ndarray of uint8, shape (H, W)
        Binary mask (0 / 255).
    phi_clip : float
        Symmetric clamp: φ is clipped to ``[-phi_clip, phi_clip]``.

    Returns
    -------
    phi : ndarray of float64, shape (H, W)
        Signed distance function: negative inside the mask, positive
        outside, clipped to the given range.
    """
    dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    return np.clip(dist_outside - dist_inside, -phi_clip, phi_clip)
