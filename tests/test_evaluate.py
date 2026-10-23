"""Tests for mask evaluation metrics (Dice and IoU)."""

import numpy as np
import pytest

from levelset_seg.evaluate import compute_dice, compute_iou, evaluate_masks


def test_identical_masks():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    assert compute_dice(mask, mask) == pytest.approx(1.0)
    assert compute_iou(mask, mask) == pytest.approx(1.0)


def test_disjoint_masks():
    m1 = np.zeros((20, 20), dtype=np.uint8)
    m2 = np.zeros((20, 20), dtype=np.uint8)
    m1[0:5, 0:5] = 1
    m2[10:15, 10:15] = 1
    assert compute_dice(m1, m2) == pytest.approx(0.0)
    assert compute_iou(m1, m2) == pytest.approx(0.0)


def test_partial_overlap():
    # m1: 10 pixels total (x from 0 to 9, y = 0)
    # m2: 10 pixels total (x from 5 to 14, y = 0)
    # intersection: 5 pixels (x from 5 to 9)
    # union: 15 pixels (x from 0 to 14)
    m1 = np.zeros((1, 20), dtype=np.uint8)
    m2 = np.zeros((1, 20), dtype=np.uint8)
    m1[0, :10] = 1
    m2[0, 5:15] = 1

    # Dice = 2 * 5 / (10 + 10) = 10 / 20 = 0.5
    # IoU = 5 / 15 = 1/3 ≈ 0.333333
    assert compute_dice(m1, m2) == pytest.approx(0.5)
    assert compute_iou(m1, m2) == pytest.approx(1.0 / 3.0)


def test_both_empty():
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10), dtype=np.uint8)
    assert compute_dice(m1, m2) == 1.0
    assert compute_iou(m1, m2) == 1.0


def test_one_empty():
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10), dtype=np.uint8)
    m1[2:5, 2:5] = 255
    assert compute_dice(m1, m2) == 0.0
    assert compute_iou(m1, m2) == 0.0


def test_shape_mismatch():
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 12), dtype=np.uint8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_dice(m1, m2)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_iou(m1, m2)


def test_evaluate_masks_dict():
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m1[2:6, 2:6] = 1
    res = evaluate_masks(m1, m1)
    assert "dice" in res
    assert "iou" in res
    assert res["dice"] == pytest.approx(1.0)
    assert res["iou"] == pytest.approx(1.0)
