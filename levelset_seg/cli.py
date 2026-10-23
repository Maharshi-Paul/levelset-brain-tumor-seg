"""Headless command-line interface for the segmentation pipeline.

Usage::

    python -m levelset_seg --input img.png --output out.png

Runs preprocessing → ROI selection → Chan-Vese solver and saves the
final contour overlay as a PNG.  No GUI or display required.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .config import SolverConfig
from .preprocessing import preprocess
from .roi import find_best_roi, initialise_phi
from .solver import evolve


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="levelset_seg",
        description=(
            "Fully automatic brain tumor segmentation from MRI. "
            "Runs the K-Means + Chan-Vese level-set pipeline "
            "without a GUI and saves the contour overlay."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input MRI image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to save the output contour-overlay PNG.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Number of PDE iterations (default: %(default)s). "
            "Overrides the config default of 85."
        ),
    )
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=None,
        help=(
            "If set, stop early when max|φ_new − φ_old| < tol. "
            "Off by default (fixed iteration count)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="OpenCV RNG seed for reproducible K-Means.",
    )
    return parser


def run_cli(args: argparse.Namespace | None = None) -> None:
    """Execute the headless segmentation pipeline."""
    parser = build_parser()
    args = parser.parse_args() if args is None else args

    # Build config, applying any CLI overrides
    overrides: dict = {}
    if args.iterations is not None:
        overrides["total_iterations"] = args.iterations
    if args.convergence_tol is not None:
        overrides["convergence_tol"] = args.convergence_tol
    cfg = SolverConfig(**overrides)

    # Optional reproducibility seed
    if args.seed is not None:
        cv2.setRNGSeed(args.seed)

    print(f"[1/4] Loading and preprocessing: {args.input}")
    image, labels, centers, _kmeans_vis, img_raw = preprocess(
        args.input, cfg
    )

    print("[2/4] Selecting ROI...")
    mask, _topology_vis, solidity = find_best_roi(
        labels, centers, img_raw.shape, cfg
    )
    if solidity > 0:
        print(f"      ROI locked (solidity={solidity:.3f})")
    else:
        print("      WARNING: No suitable ROI found, using fallback mask")

    print("[3/4] Initialising phi and running solver...")
    phi = initialise_phi(mask, cfg.phi_clip)

    def _progress(i: int, _phi: np.ndarray) -> None:
        if (i + 1) % 10 == 0 or i + 1 == cfg.total_iterations:
            print(f"      iter {i + 1}/{cfg.total_iterations}")

    phi, duration = evolve(phi, image, cfg, on_iteration=_progress)
    print(f"      Done in {duration:.3f}s")

    print(f"[4/4] Saving overlay to: {args.output}")
    # Build a colour overlay
    overlay = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

    # Draw the φ=0 contour in neon green
    phi_uint8 = ((phi > 0).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(
        phi_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (57, 255, 20), 2)  # #39ff14

    cv2.imwrite(args.output, overlay)
    print("Done.")


if __name__ == "__main__":
    run_cli()
