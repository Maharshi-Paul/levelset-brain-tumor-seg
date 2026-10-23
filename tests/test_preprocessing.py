"""Tests for the preprocessing module.

Tests cover:
- scale_image respects max_dim and preserves aspect ratio.
- enhance_contrast returns the correct shape and dtype.
- kmeans_cluster returns correctly shaped outputs.
"""

import numpy as np

from levelset_seg.preprocessing import (
    enhance_contrast,
    kmeans_cluster,
    scale_image,
)


class TestScaleImage:
    def test_downscales_large_image(self):
        """Image larger than max_dim should be downscaled."""
        img = np.zeros((800, 600), dtype=np.uint8)
        result = scale_image(img, max_dim=400)
        assert max(result.shape) <= 400

    def test_preserves_small_image(self):
        """Image already within max_dim should not be changed."""
        img = np.zeros((200, 150), dtype=np.uint8)
        result = scale_image(img, max_dim=400)
        assert result.shape == (200, 150)

    def test_aspect_ratio_preserved(self):
        """Aspect ratio should be approximately preserved."""
        img = np.zeros((800, 400), dtype=np.uint8)
        result = scale_image(img, max_dim=400)
        original_ratio = 800 / 400
        result_ratio = result.shape[0] / result.shape[1]
        assert abs(original_ratio - result_ratio) < 0.1


class TestEnhanceContrast:
    def test_output_shape(self):
        """Output shape should match input shape."""
        img = np.random.randint(0, 255, (100, 80), dtype=np.uint8)
        result = enhance_contrast(img, 2.0, (8, 8), (5, 5))
        assert result.shape == img.shape

    def test_output_dtype(self):
        """Output dtype should be uint8."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = enhance_contrast(img, 2.0, (8, 8), (5, 5))
        assert result.dtype == np.uint8


class TestKmeansCluster:
    def test_output_shapes(self):
        """K-Means should return correctly shaped outputs."""
        img = np.random.randint(0, 255, (100, 80), dtype=np.uint8)
        labels, centers, vis = kmeans_cluster(img, k=5, max_iter=100, epsilon=0.2, attempts=3)

        assert labels.shape == (100 * 80, 1)
        assert centers.shape == (5, 1)
        assert vis.shape == img.shape

    def test_labels_in_range(self):
        """All labels should be in [0, k-1]."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        labels, _, _ = kmeans_cluster(img, k=5, max_iter=100, epsilon=0.2, attempts=3)

        assert labels.min() >= 0
        assert labels.max() < 5

    def test_vis_dtype(self):
        """Visualisation image should be uint8."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        _, _, vis = kmeans_cluster(img, k=3, max_iter=50, epsilon=0.2, attempts=2)
        assert vis.dtype == np.uint8
