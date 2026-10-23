"""Evaluation metrics for binary segmentation masks.

Computes standard overlap metrics:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard Index)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

import cv2
import numpy as np
from numpy.typing import NDArray


def compute_dice(
    pred: NDArray[np.generic],
    gt: NDArray[np.generic],
) -> float:
    """Compute the Dice Similarity Coefficient (DSC) between two binary masks.

    .. math::
        \\text{DSC} = \\frac{2 |P \\cap G|}{|P| + |G|}

    Parameters
    ----------
    pred : ndarray
        Predicted binary mask (non-zero is foreground).
    gt : ndarray
        Ground-truth binary mask (non-zero is foreground).

    Returns
    -------
    float
        Dice score in [0.0, 1.0]. Returns 1.0 if both masks are empty.

    Raises
    ------
    ValueError
        If *pred* and *gt* shapes do not match.
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred shape {pred.shape} != gt shape {gt.shape}"
        )

    p = pred != 0
    g = gt != 0

    p_sum = int(np.count_nonzero(p))
    g_sum = int(np.count_nonzero(g))

    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0

    intersection = int(np.count_nonzero(p & g))
    return float((2.0 * intersection) / (p_sum + g_sum))


def compute_iou(
    pred: NDArray[np.generic],
    gt: NDArray[np.generic],
) -> float:
    """Compute the Intersection over Union (IoU / Jaccard Index).

    .. math::
        \\text{IoU} = \\frac{|P \\cap G|}{|P \\cup G|}

    Parameters
    ----------
    pred : ndarray
        Predicted binary mask (non-zero is foreground).
    gt : ndarray
        Ground-truth binary mask (non-zero is foreground).

    Returns
    -------
    float
        IoU score in [0.0, 1.0]. Returns 1.0 if both masks are empty.

    Raises
    ------
    ValueError
        If *pred* and *gt* shapes do not match.
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred shape {pred.shape} != gt shape {gt.shape}"
        )

    p = pred != 0
    g = gt != 0

    intersection = int(np.count_nonzero(p & g))
    union = int(np.count_nonzero(p | g))

    if union == 0:
        return 1.0

    return float(intersection / union)


def evaluate_masks(
    pred: NDArray[np.generic],
    gt: NDArray[np.generic],
) -> Dict[str, float]:
    """Evaluate multiple overlap metrics between two masks."""
    return {
        "dice": compute_dice(pred, gt),
        "iou": compute_iou(pred, gt),
    }


def main() -> None:
    """Command-line entry point for mask evaluation."""
    parser = argparse.ArgumentParser(
        description="Compute Dice and IoU metrics between predicted and ground-truth binary masks."
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to predicted binary mask image.",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to ground-truth binary mask image.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output metrics as a JSON string.",
    )
    args = parser.parse_args()

    pred_img = cv2.imread(args.pred, cv2.IMREAD_GRAYSCALE)
    if pred_img is None:
        print(f"Error: Could not load prediction mask from {args.pred}", file=sys.stderr)
        sys.exit(1)

    gt_img = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        print(f"Error: Could not load ground-truth mask from {args.gt}", file=sys.stderr)
        sys.exit(1)

    try:
        metrics = evaluate_masks(pred_img, gt_img)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(f"Dice Coefficient: {metrics['dice']:.4f}")
        print(f"IoU (Jaccard):    {metrics['iou']:.4f}")


if __name__ == "__main__":
    main()
