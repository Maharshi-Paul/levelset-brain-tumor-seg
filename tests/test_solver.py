"""Tests for the Chan-Vese level-set solver.

Tests cover:
- Regularised Heaviside behaviour at phi=0 and large |phi|.
- Regularised Dirac behaviour at phi=0 and large |phi|.
- Full solver on a synthetic bright-disc image: the phi=0 contour
  should end up close to the disc boundary (IoU >= 0.7).
"""

import numpy as np
import pytest

from levelset_seg.config import SolverConfig
from levelset_seg.solver import (
    compute_pde_step,
    evolve,
    regularized_dirac,
    regularized_heaviside,
)

# ── Heaviside tests ─────────────────────────────────────────────────


class TestHeaviside:
    def test_at_zero(self):
        """H_eps(0) should equal 0.5 for any epsilon."""
        assert regularized_heaviside(0.0, 1.0) == pytest.approx(0.5)
        assert regularized_heaviside(0.0, 0.1) == pytest.approx(0.5)

    def test_large_positive(self):
        """H_eps(large) -> 1.0."""
        assert regularized_heaviside(1000.0, 1.0) == pytest.approx(1.0, abs=1e-3)

    def test_large_negative(self):
        """H_eps(-large) -> 0.0."""
        assert regularized_heaviside(-1000.0, 1.0) == pytest.approx(0.0, abs=1e-3)

    def test_monotonically_increasing(self):
        """H_eps should be monotonically increasing in phi."""
        phi_values = np.linspace(-5, 5, 100)
        h_values = [regularized_heaviside(p, 1.0) for p in phi_values]
        for i in range(1, len(h_values)):
            assert h_values[i] >= h_values[i - 1]


# ── Dirac tests ─────────────────────────────────────────────────────


class TestDirac:
    def test_at_zero(self):
        """delta_eps(0) = eps / (pi * eps^2) = 1/pi when eps=1."""
        expected = 1.0 / np.pi
        assert regularized_dirac(0.0, 1.0) == pytest.approx(expected)

    def test_large_phi(self):
        """delta_eps(large) -> 0."""
        assert regularized_dirac(1000.0, 1.0) == pytest.approx(0.0, abs=1e-4)
        assert regularized_dirac(-1000.0, 1.0) == pytest.approx(0.0, abs=1e-4)

    def test_symmetric(self):
        """delta_eps should be symmetric: delta(phi) == delta(-phi)."""
        for phi in [0.5, 1.0, 2.0, 10.0]:
            assert regularized_dirac(phi, 1.0) == pytest.approx(
                regularized_dirac(-phi, 1.0)
            )

    def test_peak_at_zero(self):
        """delta_eps(0) should be the maximum value."""
        peak = regularized_dirac(0.0, 1.0)
        for phi in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
            assert regularized_dirac(phi, 1.0) <= peak


# ── Solver integration test with synthetic disc ─────────────────────


def _make_disc_image(
    size: int = 100,
    radius: int = 20,
    fg_intensity: float = 0.8,
    bg_intensity: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic image with a bright disc on a dark background.

    Returns (image, ground_truth_mask) where mask is True inside the disc.
    """
    y, x = np.mgrid[:size, :size]
    centre = size // 2
    dist = np.sqrt((x - centre) ** 2 + (y - centre) ** 2)
    mask = dist <= radius
    image = np.full((size, size), bg_intensity, dtype=np.float64)
    image[mask] = fg_intensity
    return image, mask


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Intersection over Union between two boolean masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


class TestSolverDisc:
    """End-to-end solver test on a synthetic bright disc."""

    def test_disc_segmentation_iou(self):
        """The phi=0 contour should closely match the disc boundary."""
        image, gt_mask = _make_disc_image(size=100, radius=20)

        # Initialise phi as a signed distance from a slightly larger disc
        # (simulating an imperfect initial ROI)
        y, x = np.mgrid[:100, :100]
        init_radius = 25  # deliberately larger than true radius
        dist = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        phi_init = dist - init_radius  # positive outside, negative inside
        phi_init = np.clip(phi_init, -5.0, 5.0)

        cfg = SolverConfig(
            total_iterations=85,
            dt=0.5,
            lambda1=1.0,
            lambda2=1.0,
            epsilon=1.0,
        )
        # Image variance is low for this synthetic image, so mu=0.5
        phi_final, _duration = evolve(phi_init, image, cfg)

        # phi < 0 is "inside" the contour
        pred_mask = phi_final < 0
        iou = _compute_iou(pred_mask, gt_mask)

        assert iou >= 0.7, f"IoU {iou:.3f} is below threshold 0.7"

    def test_single_step_does_not_crash(self):
        """compute_pde_step should not crash on a small image."""
        image = np.random.rand(10, 10).astype(np.float64)
        phi = np.zeros((10, 10), dtype=np.float64)
        result = compute_pde_step(phi, image, 0.25, 1.0, 1.0, 0.5, 1.0)
        assert result.shape == (10, 10)
        assert np.isfinite(result).all()
