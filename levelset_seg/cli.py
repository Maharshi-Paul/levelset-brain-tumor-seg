"""Headless command-line interface for the segmentation pipeline.

Supports single-image and batch-directory processing, optional panel visualisations,
and JSON metrics output.

Usage::

    # Single image
    python -m levelset_seg --input data/Te-me_0015.jpg --output out.png

    # Batch directory with 3-panel visualisations and metrics
    python -m levelset_seg --input data/ --output results/ --save-panels --json metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .config import SolverConfig
from .preprocessing import preprocess
from .roi import find_best_roi, initialise_phi
from .solver import evolve

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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
        help="Path to an input MRI image or a directory of images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to save output overlay PNG (or directory if --input is a directory).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of PDE iterations (overrides config default of 85).",
    )
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=None,
        help="Stop early when max|phi_new - phi_old| < tol (off by default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="OpenCV RNG seed for reproducible K-Means.",
    )
    parser.add_argument(
        "--save-panels",
        action="store_true",
        help="Save side-by-side 3-panel visualization (K-Means, ROI, level set).",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="FILE",
        help="Output execution metrics as JSON (pass filename, or omit to print to stdout).",
    )
    return parser


def process_image(
    input_path: Path,
    output_overlay_path: Path,
    cfg: SolverConfig,
    save_panels: bool = False,
    panels_output_path: Path | None = None,
    seed: int | None = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Process a single MRI image through the full segmentation pipeline."""
    if seed is not None:
        cv2.setRNGSeed(seed)

    if not quiet:
        print(f"[1/4] Loading and preprocessing: {input_path}")
    image, labels, centers, kmeans_vis, img_raw = preprocess(str(input_path), cfg)

    if not quiet:
        print("[2/4] Selecting ROI...")
    mask, topology_vis, solidity = find_best_roi(labels, centers, img_raw.shape, cfg)
    if not quiet:
        if solidity > 0:
            print(f"      ROI locked (solidity={solidity:.3f})")
        else:
            print("      WARNING: No suitable ROI found, using fallback mask")

    if not quiet:
        print("[3/4] Initialising phi and running solver...")
    phi = initialise_phi(mask, cfg.phi_clip)

    def _progress(i: int, _phi: np.ndarray) -> None:
        if not quiet and ((i + 1) % 10 == 0 or i + 1 == cfg.total_iterations):
            print(f"      iter {i + 1}/{cfg.total_iterations}")

    phi, duration = evolve(phi, image, cfg, on_iteration=_progress)
    if not quiet:
        print(f"      Done in {duration:.3f}s")

    if not quiet:
        print(f"[4/4] Saving overlay to: {output_overlay_path}")
    output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

    # Draw the phi=0 boundary contour in neon green (#39ff14 -> BGR: 20, 255, 57)
    phi_uint8 = ((phi > 0).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(
        phi_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (57, 255, 20), 2)
    cv2.imwrite(str(output_overlay_path), overlay)

    if save_panels and panels_output_path is not None:
        panels_output_path.parent.mkdir(parents=True, exist_ok=True)
        p1 = cv2.applyColorMap(kmeans_vis, cv2.COLORMAP_MAGMA)
        p2 = topology_vis
        p3 = overlay
        panels = np.hstack([p1, p2, p3])
        cv2.imwrite(str(panels_output_path), panels)

    return {
        "input": str(input_path),
        "output": str(output_overlay_path),
        "panels": str(panels_output_path) if save_panels else None,
        "shape": list(img_raw.shape),
        "roi_solidity": round(float(solidity), 4),
        "duration_seconds": round(float(duration), 4),
        "variance": round(float(np.var(image)), 6),
        "iterations": cfg.total_iterations,
    }


def run_cli(args: argparse.Namespace | None = None) -> None:
    """Execute the headless segmentation pipeline."""
    parser = build_parser()
    args = parser.parse_args() if args is None else args

    overrides: Dict[str, Any] = {}
    if args.iterations is not None:
        overrides["total_iterations"] = args.iterations
    if args.convergence_tol is not None:
        overrides["convergence_tol"] = args.convergence_tol
    cfg = SolverConfig(**overrides)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    all_metrics: List[Dict[str, Any]] = []

    if in_path.is_dir():
        image_files = sorted(
            [p for p in in_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if not image_files:
            print(
                f"Error: No image files found in '{args.input}'.", file=sys.stderr
            )
            sys.exit(1)

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Batch processing {len(image_files)} images from {in_path} to {out_dir}...")

        for idx, img_file in enumerate(image_files, start=1):
            print(f"\n--- [{idx}/{len(image_files)}] {img_file.name} ---")
            out_file = out_dir / f"{img_file.stem}_seg.png"
            panels_file = (
                out_dir / f"{img_file.stem}_panels.png" if args.save_panels else None
            )
            metrics = process_image(
                input_path=img_file,
                output_overlay_path=out_file,
                cfg=cfg,
                save_panels=args.save_panels,
                panels_output_path=panels_file,
                seed=args.seed,
            )
            all_metrics.append(metrics)
    else:
        out_path = Path(args.output)
        if out_path.is_dir() or str(args.output).endswith(("/", "\\")):
            out_file = out_path / f"{in_path.stem}_seg.png"
        else:
            out_file = out_path

        panels_file = (
            out_file.parent / f"{out_file.stem}_panels.png"
            if args.save_panels
            else None
        )
        metrics = process_image(
            input_path=in_path,
            output_overlay_path=out_file,
            cfg=cfg,
            save_panels=args.save_panels,
            panels_output_path=panels_file,
            seed=args.seed,
        )
        all_metrics.append(metrics)

    if args.json is not None:
        result_payload = all_metrics if len(all_metrics) > 1 else all_metrics[0]
        json_str = json.dumps(result_payload, indent=2)
        if args.json == "-":
            print("\n=== METRICS JSON ===")
            print(json_str)
        else:
            json_file = Path(args.json)
            json_file.parent.mkdir(parents=True, exist_ok=True)
            json_file.write_text(json_str, encoding="utf-8")
            print(f"Metrics saved to: {json_file}")

    print("\nCompleted successfully.")


if __name__ == "__main__":
    run_cli()
