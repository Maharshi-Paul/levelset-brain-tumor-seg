# levelset-brain-tumor-seg

Multi-phase brain tumor segmentation from MRI using level set PDEs (active contours without edges), with a live CustomTkinter dashboard and a Numba-parallelized PDE solver.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

The app automatically segments tumor regions from grayscale MRI slices with no manual seeding required. It runs a three-stage pipeline and visualizes every stage live:

1. **Unsupervised clustering** — K-Means (5 clusters) on a CLAHE-enhanced, denoised image to separate tissue intensity classes.
2. **Topological ROI search** — the three brightest clusters are scanned for contours, and each candidate is scored on area, solidity (convexity), and circularity to automatically select the most tumor-like region — no manual seed point needed.
3. **Level set evolution** — the selected ROI is converted into a signed distance function and evolved with a Chan-Vese–style (active contours without edges) PDE, solved with a regularized Heaviside/Dirac formulation combining a curvature term and a two-phase region-fitting term.

All three stages are rendered side-by-side in real time as the simulation runs.

## Features

- **Numba JIT + parallel (`prange`) PDE solver** for fast, multi-core evolution of the level set
- **Adaptive curvature weight (`mu`)** — chosen automatically based on image variance
- **Automatic ROI selection** via contour scoring (area × solidity³ × circularity³), removing the need for manual initialization
- **Live 3-panel visualization**: K-Means clusters → convex-hull ROI → evolving `φ = 0` contour, refreshed every 10 iterations
- **Batch processing** with Prev/Next navigation across multiple uploaded images
- **Dark-themed desktop GUI** (CustomTkinter) with progress bar, status indicator, and a running console log
- **Preprocessing pipeline**: grayscale conversion, downscaling (max dimension 400px for solver performance), CLAHE contrast equalization, Gaussian smoothing

## Mathematical Background

The segmentation boundary is represented implicitly as the zero level set of a signed distance function `φ`. At each iteration:

- Regional means `c1` (inside) and `c2` (outside) are computed as Heaviside-weighted averages of image intensity.
- `φ` is updated according to:

  ```
  ∂φ/∂t = δ(φ) · [ μ·κ(φ)  −  λ1·(I − c1)²  +  λ2·(I − c2)² ]
  ```

  where `κ` is the curvature of the level set (computed from first/second-order finite differences), and `δ`, the regularized Dirac delta, and `H`, the regularized Heaviside function, are smoothed approximations for numerical stability.

This is the classic **Chan-Vese "active contours without edges"** formulation, which segments regions by intensity homogeneity rather than by gradient/edge strength — well suited to MRI, where tumor boundaries are often low-contrast.

**Default parameters** (tunable in code):

| Parameter | Value | Description |
|---|---|---|
| `total_iterations` | 85 | Number of PDE evolution steps |
| `dt` | 0.5 | Time step |
| `λ1`, `λ2` | 1.0, 1.0 | Inside/outside region fitting weights |
| `μ` | 0.5 or 0.25 | Curvature weight (adaptive on image variance) |
| `ε` | 1.0 | Heaviside/Dirac regularization width |

## Tech Stack

- Python 3
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/) (`opencv-python`) — I/O, CLAHE, K-Means, contours, distance transform
- [Numba](https://numba.pydata.org/) — JIT compilation + parallel loops (`prange`) for the PDE core
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) — GUI
- [Matplotlib](https://matplotlib.org/) — embedded visualization canvas

## Installation

```bash
pip install numpy opencv-python customtkinter matplotlib numba
```

> Requires Python 3.9+. Numba's JIT compilation may take a few seconds on first run.

## Usage

```bash
python main.py
```

1. Click **Upload Batch & Segment** and select one or more MRI images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.bmp`).
2. The pipeline runs automatically: clustering → ROI detection → level set initialization → PDE evolution.
3. Watch the live evolution of the segmentation boundary in the third panel.
4. Once complete, use **Prev / Next** to step through the batch — each image is segmented independently.

## How It Works, Panel by Panel

| Panel | Shows |
|---|---|
| 1. Unsupervised Clusters | Raw K-Means output over pixel intensities |
| 2. Convex Hull ROI | The auto-selected tumor candidate contour (cyan fill) and its convex hull (green outline), with a live solidity score |
| 3. Level Set Evolution | The MRI slice with the evolving `φ = 0` contour (neon green) overlaid, updated every 10 iterations until convergence |

## Limitations

- Operates on 2D slices; does not perform full 3D volumetric segmentation.
- Automatic ROI selection assumes the tumor is among the brighter intensity clusters — atypical or very low-contrast lesions may need manual tuning of `search_clusters` or the scoring heuristic.
- Research/educational tool — **not validated for clinical or diagnostic use**.

## License

MIT — see [LICENSE](LICENSE) for details.