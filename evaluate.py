#!/usr/bin/env python
"""CLI tool to compute Dice and IoU metrics between predicted and ground-truth masks."""

from levelset_seg.evaluate import compute_dice, compute_iou, evaluate_masks, main

__all__ = ["compute_dice", "compute_iou", "evaluate_masks", "main"]

if __name__ == "__main__":
    main()
