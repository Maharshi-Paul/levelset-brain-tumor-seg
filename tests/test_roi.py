"""Tests for the automatic ROI selection module.

Tests cover:
- ROI finder correctly selects a bright disc on a noisy dark background.
- initialise_phi produces a valid SDF with correct sign convention.
"""

import cv2
import numpy as np

from levelset_seg.config import SolverConfig
from levelset_seg.roi import find_best_roi, initialise_phi


def _make_disc_kmeans_data(
    size: int = 100,
    radius: int = 20,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Simulate K-Means output for a bright disc on dark background.

    Returns (labels, centers, image_shape) matching the interface
    expected by find_best_roi.
    """
    y, x = np.mgrid[:size, :size]
    dist = np.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2)
    mask = dist <= radius

    # Simulate 3-cluster K-Means: dark bg (0), medium ring (1), bright disc (2)
    labels = np.zeros(size * size, dtype=np.int32)
    labels[mask.flatten()] = 2

    # Add a thin medium-intensity ring around the disc
    ring = (dist > radius) & (dist <= radius + 5)
    labels[ring.flatten()] = 1

    labels = labels.reshape(-1, 1)
    centers = np.array([[30.0], [128.0], [220.0]], dtype=np.float32)

    return labels, centers, (size, size)


class TestFindBestROI:
    def test_selects_disc(self):
        """ROI finder should select the bright disc region."""
        labels, centers, shape = _make_disc_kmeans_data(size=100, radius=20)
        cfg = SolverConfig(
            kmeans_k=3,
            top_n_clusters=3,
            min_contour_area=50,
        )

        mask, _topo_vis, solidity = find_best_roi(
            labels, centers, shape, cfg
        )

        # mask should be non-empty
        assert mask.sum() > 0, "ROI mask is empty"
        # Solidity should be high for a disc
        assert solidity > 0.9, f"Solidity {solidity:.3f} too low for a disc"

    def test_mask_dtype_and_range(self):
        """ROI mask should be uint8 with values 0 or 255."""
        labels, centers, shape = _make_disc_kmeans_data()
        cfg = SolverConfig(kmeans_k=3, top_n_clusters=3)
        mask, _, _ = find_best_roi(labels, centers, shape, cfg)

        assert mask.dtype == np.uint8
        unique = set(np.unique(mask))
        assert unique <= {0, 255}

    def test_topology_vis_shape(self):
        """Topology visualisation should be a 3-channel colour image."""
        labels, centers, shape = _make_disc_kmeans_data()
        cfg = SolverConfig(kmeans_k=3, top_n_clusters=3)
        _, topo_vis, _ = find_best_roi(labels, centers, shape, cfg)

        assert topo_vis.ndim == 3
        assert topo_vis.shape[:2] == shape
        assert topo_vis.shape[2] == 3


class TestInitialisePhi:
    def test_sign_convention(self):
        """phi should be negative inside the mask, positive outside."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(mask, (25, 25), 10, 255, thickness=cv2.FILLED)
        phi = initialise_phi(mask, phi_clip=5.0)

        # Centre of disc should be negative
        assert phi[25, 25] < 0
        # Corner (far outside) should be positive
        assert phi[0, 0] > 0

    def test_clipping(self):
        """phi values should be within [-phi_clip, phi_clip]."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(mask, (25, 25), 10, 255, thickness=cv2.FILLED)

        clip = 3.0
        phi = initialise_phi(mask, phi_clip=clip)

        assert phi.min() >= -clip
        assert phi.max() <= clip

    def test_dtype(self):
        """phi should be floating point."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        phi = initialise_phi(mask, phi_clip=5.0)
        assert np.issubdtype(phi.dtype, np.floating)
