<p align="center">
  <img src="assets/levelset_banner.png" alt="levelset-brain-tumor-seg banner" width="100%">
</p>

# Level-Set Brain Tumor Segmentation

<p align="center">
  <a href="https://github.com/Maharshi-Paul/levelset-brain-tumor-seg/actions/workflows/ci.yml"><img src="https://github.com/Maharshi-Paul/levelset-brain-tumor-seg/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg" alt="Python Versions">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/compute-CPU--only%20(Numba%20JIT)-orange.svg" alt="Compute: CPU-only">
</p>

**Fully automatic brain tumor segmentation from MRI — zero manual seeding, zero training data, and zero GPU required.** An unsupervised K-Means and topological ROI search pipeline feeds a Numba CPU-parallel Chan-Vese level-set PDE solver, visualized live in a CustomTkinter desktop dashboard or executed headlessly via CLI (reproducible with `--seed`).

Built by [Maharshi-Paul](https://github.com/Maharshi-Paul)  
Poster design by [DIPESHCHOUDHARY-CODES](https://github.com/DIPESHCHOUDHARY-CODES)

---

## Quick Start

Get running in three commands:

```bash
# 1. Clone repository
git clone https://github.com/Maharshi-Paul/levelset-brain-tumor-seg.git
cd levelset-brain-tumor-seg

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch interactive GUI (or run headlessly via CLI)
python main.py
# Headless CLI alternative:
# python -m levelset_seg -i data/ -o results/ --save-panels --json metrics.json
```

---

## Screenshots & Demo

<!--
TODO: Place real dashboard screenshots and recordings in assets/ once captured:
- assets/dashboard.png (Full 3-panel GUI view during or after segmentation)
- assets/demo.gif (Animated screen recording of contour convergence)

| Full Application Dashboard | Dynamic Contour Evolution |
|:---:|:---:|
| ![Dashboard Screenshot](assets/dashboard.png) | ![Segmentation Demo GIF](assets/demo.gif) |
-->

*Interactive 3-panel dashboard view: Panel 1 displays unsupervised K-Means intensity clustering; Panel 2 isolates the candidate tumor contour with its green convex hull and live solidity metric; Panel 3 renders the evolving neon-green level-set boundary over the MRI slice in real time.*

---

## Table of Contents

- [Quick Start](#quick-start)
- [Screenshots & Demo](#screenshots--demo)
- [Problem](#problem)
- [Solution](#solution)
- [On-Device Compute Usage](#on-device-compute-usage)
- [Mathematical Formulation](#mathematical-formulation)
- [Project Architecture](#project-architecture)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Interactive GUI Dashboard](#1-interactive-gui-dashboard)
  - [Headless Command-Line Interface (CLI)](#2-headless-command-line-interface-cli)
  - [Mask Evaluation Tool](#3-mask-evaluation-tool-evaluatepy)
- [Running Tests & Linting](#running-tests--linting)
- [How It Works, Panel by Panel](#how-it-works-panel-by-panel)
- [Experimental Results](#experimental-results)
- [Data Source & Attribution](#data-source--attribution)
- [Performance Optimizations](#performance-optimizations)
- [Clinical Disclaimer & Limitations](#clinical-disclaimer--limitations)
- [References](#references)
- [Citation](#citation)
- [License](#license)

---

## Problem

Brain tumor segmentation in magnetic resonance imaging (MRI) is traditionally addressed in two ways:
1. **Manual tracing by a radiologist** — accurate but labour-intensive, subjective, and difficult to scale across clinical workflows.
2. **Deep learning models** — highly capable but dependent on massive pixel-annotated datasets, heavy GPU infrastructure, and susceptible to distribution shift across different scanners, field strengths, and acquisition protocols.

Simpler intensity thresholding breaks down on low-contrast, noisy tissue margins. Classical active-contour (snake) methods require an expert human operator to manually specify a seed point or bounding box inside the lesion before evolution can begin.

---

## Solution

`levelset-brain-tumor-seg` eliminates manual intervention entirely. It provides an end-to-end, CPU-only pipeline (reproducible with `--seed`):

```
[MRI Slice] ──> [CLAHE & Blur] ──> [K-Means (k=5)] ──> [Topological ROI Scoring] ──> [Initial SDF φ₀] ──> [Chan-Vese PDE (Numba CPU)] ──> [Tumor Contour]
```

1. **Contrast Enhancement & Unsupervised Clustering**: Contrast-Limited Adaptive Histogram Equalization (CLAHE) and mild Gaussian filtering enhance tissue contrast, followed by OpenCV K-Means ($k=5$) to partition intensities without supervision.
2. **Topological ROI Discovery**: The brightest tissue clusters are scanned for candidate contours. Each contour is evaluated using a geometric scoring heuristic balancing area, convexity (solidity), and circularity:
   $$\text{Score} = \sqrt{\text{Area}} \cdot \text{Solidity}^3 \cdot \text{Circularity}^3$$
   This automatically isolates round, convex tumor masses without seed points.
3. **Two-Phase Chan-Vese Level-Set PDE**: The detected ROI initializes a signed distance function (SDF) $\phi_0$, which is evolved according to the Chan-Vese energy functional using CPU-parallelized finite differences to lock onto the precise tissue boundary.

> **Note on Terminology**: The overall *software pipeline* operates in multiple stages (preprocessing $\rightarrow$ topological selection $\rightarrow$ PDE evolution), while the mathematical level-set solver implements a **two-phase** Chan-Vese model (partitioning the domain into interior $\phi < 0$ and exterior $\phi > 0$ regions).

---

## On-Device Compute Usage

**Runs 100% locally on standard consumer CPUs — no cloud dependencies, no GPU required:**

| Component | Operation | Engine |
|---|---|---|
| Image Preprocessing | CLAHE, Gaussian blur, dimension scaling | OpenCV (`cv2`) |
| Unsupervised Clustering | Intensity grouping into 5 clusters | OpenCV K-Means (CPU) |
| Topological Search | Contour extraction, convex hull, shape scoring | OpenCV Contours |
| SDF Initialization | Exact Euclidean Distance Transform | OpenCV Distance Transform ($L_2$) |
| Level-Set PDE Solver | Curvature, Dirac/Heaviside updates, reductions | Numba JIT + `prange` CPU parallelism (`cache=True`) |

No network telemetry, no external API calls, and no patient data leaves the local machine.

---

<details>
<summary><strong>Mathematical Formulation (Click to expand)</strong></summary>

### Level-Set Representation and Sign Convention

The tumor boundary $C$ is represented implicitly as the zero level set of a Lipschitz-continuous signed distance function $\phi: \Omega \rightarrow \mathbb{R}$:

$$C = \{ \mathbf{x} \in \Omega \mid \phi(\mathbf{x}) = 0 \}$$

In accordance with the solver implementation:
- **Tumor interior**: $\phi(\mathbf{x}) < 0$
- **Exterior / background**: $\phi(\mathbf{x}) > 0$
- **Interface / contour**: $\phi(\mathbf{x}) = 0$

### Regional Means

Regional means $c_1$ and $c_2$ are computed via continuous Heaviside weighting:
- $c_1$ is the average intensity of the **exterior / background** region ($\phi > 0$), weighted by the smoothed Heaviside step $H_\varepsilon(\phi)$:
  $$c_1 = \frac{\int_\Omega I(\mathbf{x}) \cdot H_\varepsilon(\phi(\mathbf{x})) \, d\mathbf{x}}{\int_\Omega H_\varepsilon(\phi(\mathbf{x})) \, d\mathbf{x}}$$
- $c_2$ is the average intensity of the **tumor interior** region ($\phi < 0$), weighted by $1 - H_\varepsilon(\phi)$:
  $$c_2 = \frac{\int_\Omega I(\mathbf{x}) \cdot (1 - H_\varepsilon(\phi(\mathbf{x}))) \, d\mathbf{x}}{\int_\Omega (1 - H_\varepsilon(\phi(\mathbf{x}))) \, d\mathbf{x}}$$

### Chan-Vese Energy Functional

The two-phase Chan-Vese functional (Chan & Vese, 2001) being minimized is:

$$F(c_1, c_2, \phi) = \mu \cdot \text{Length}(\phi = 0) + \lambda_1 \int_\Omega |I(\mathbf{x}) - c_1|^2 H_\varepsilon(\phi) \, d\mathbf{x} + \lambda_2 \int_\Omega |I(\mathbf{x}) - c_2|^2 (1 - H_\varepsilon(\phi)) \, d\mathbf{x}$$

where:
- $I(\mathbf{x}) \in [0, 1]$ is the normalized image intensity.
- $\mu$ controls boundary smoothness (penalizing contour length).
- $\lambda_1, \lambda_2$ scale exterior and interior data fidelity.

### Euler-Lagrange Evolution PDE

Gradient descent on $F$ with respect to $\phi$ yields the evolution equation implemented in `compute_pde_step`:

$$\frac{\partial \phi}{\partial t} = \delta_\varepsilon(\phi) \left[ \mu \cdot \kappa(\phi) - \lambda_1 (I - c_1)^2 + \lambda_2 (I - c_2)^2 \right]$$

where $\kappa(\phi) = \text{div}\left(\frac{\nabla \phi}{|\nabla \phi|}\right)$ is the mean curvature computed using central finite differences:

$$\kappa = \frac{\phi_{xx} \phi_y^2 - 2 \phi_x \phi_y \phi_{xy} + \phi_{yy} \phi_x^2}{(\phi_x^2 + \phi_y^2 + \eta)^{3/2}}$$

Here $\eta = 10^{-10}$ is a numerical regularization constant in the denominator preventing division by zero in zero-gradient flat regions where $|\nabla \phi| \approx 0$.

### Regularized Heaviside and Dirac Delta

To ensure smooth numerical differentiation across the interface, $H(\phi)$ and $\delta(\phi)$ are regularized with parameter $\varepsilon = 1.0$:

$$H_\varepsilon(\phi) = \frac{1}{2} \left( 1 + \frac{2}{\pi} \arctan\left(\frac{\phi}{\varepsilon}\right) \right)$$

$$\delta_\varepsilon(\phi) = \frac{\varepsilon / \pi}{\varepsilon^2 + \phi^2}$$

### Default Solver Parameters

All defaults are defined in `SolverConfig` (`levelset_seg/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `total_iterations` | `85` | Fixed evolution steps (when convergence stopping is inactive) |
| `dt` | `0.5` | Euler integration time step |
| `lambda1`, `lambda2` | `1.0`, `1.0` | Exterior and interior data-fidelity weights |
| `mu` | Adaptive (`0.25` or `0.5`) | Curvature weight: $0.25$ if $\text{Var}(I) \ge 0.05$ (textured/noisy), else $0.5$ (smooth) |
| `variance_threshold` | `0.05` | Variance cutoff determining adaptive $\mu$ |
| `epsilon` | `1.0` | Heaviside / Dirac regularization width |
| `max_dim` | `400` | Maximum spatial dimension for aspect-preserving scaling |
| `kmeans_k` | `5` | Number of clusters for intensity partitioning |
| `phi_clip` | `5.0` | Symmetric signed distance clamp $[-5.0, +5.0]$ |
| `convergence_tol` | `None` | Optional early stopping tolerance $\max|\phi_{t+1} - \phi_t| < \text{tol}$ |

</details>

---

## Project Architecture

```
levelset-brain-tumor-seg/
├── levelset_seg/                  # Core package
│   ├── __init__.py                # Package export & version metadata
│   ├── __main__.py                # CLI entry point (python -m levelset_seg)
│   ├── cli.py                     # Headless CLI (batch, panels, JSON)
│   ├── config.py                  # SolverConfig dataclass (single source of truth)
│   ├── evaluate.py                # Dice and IoU evaluation functions
│   ├── gui.py                     # CustomTkinter + Matplotlib dashboard
│   ├── preprocessing.py          # CLAHE, Gaussian blur, scaling, K-Means
│   ├── roi.py                     # Topological contour scoring & SDF creation
│   └── solver.py                  # Numba-parallelized Chan-Vese PDE solver (cache=True)
├── tests/                         # Pytest test suite
│   ├── __init__.py
│   ├── test_cli.py                # Headless CLI & batch tests
│   ├── test_evaluate.py           # Dice & IoU unit tests on synthetic masks
│   ├── test_preprocessing.py      # CLAHE & K-Means unit tests
│   ├── test_roi.py                # Shape scoring & SDF clamp tests
│   └── test_solver.py             # Math invariants (Heaviside/Dirac) & synthetic IoU
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI matrix (Python 3.10–3.14)
├── assets/                        # Visual banners & UI assets
│   └── levelset_banner.png
├── data/                          # Sample brain MRI slices (meningioma)
├── evaluate.py                    # Standalone evaluation CLI
├── main.py                        # Thin application launcher
├── pyproject.toml                 # Ruff linting & pytest configuration
├── requirements.txt               # Pinned core production dependencies
├── requirements-dev.txt           # Testing & linting tools
├── CITATION.cff                   # Citation metadata
├── .gitignore
├── LICENSE                        # MIT License
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python $\ge$ 3.10 and $\le$ 3.14 (tested on 3.10, 3.11, 3.12, 3.13, and 3.14.3).
- Works natively on Windows, macOS, and Linux.

### 1. Clone the Repository

```bash
git clone https://github.com/Maharshi-Paul/levelset-brain-tumor-seg.git
cd levelset-brain-tumor-seg
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Note for Linux / macOS GUI Users**: If running the graphical desktop interface on Linux or macOS, ensure system Tkinter bindings are present:
> - **Ubuntu / Debian**: `sudo apt-get install python3-tk`
> - **macOS (Homebrew)**: `brew install python-tk`

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To install development and testing tools:
```bash
pip install -r requirements-dev.txt
```

> **Note on Numba Compilation**: Numba JIT functions use `cache=True`. Compilation occurs on the initial invocation and is cached for subsequent runs, delivering near-native execution speed.

---

## Usage

### 1. Interactive GUI Dashboard

Launch the desktop application:

```bash
python main.py
```

1. Click **Upload Batch & Segment** and choose one or more MRI images from `data/` or your own files (`.png`, `.jpg`, `.jpeg`, `.tif`, `.bmp`).
2. Watch the three-panel display:
   - Panel 1 shows unsupervised intensity clustering.
   - Panel 2 isolates the candidate tumor contour and highlights its convex hull with live solidity metrics.
   - Panel 3 visualizes the real-time PDE contour evolution overlaid on the MRI slice.
3. Use the **Prev** and **Next** navigation buttons to step through segmented images in the batch.

### 2. Headless Command-Line Interface (CLI)

For batch processing, automated scripting, or headless servers (without Tkinter/X11):

```bash
# Basic single-image segmentation
python -m levelset_seg --input data/Te-me_0015.jpg --output output_overlay.png

# Batch-process an entire directory of images
python -m levelset_seg --input data/ --output results/

# Full batch run with 3-panel visualisations, metrics JSON, and fixed RNG seed:
python -m levelset_seg -i data/ -o results/ --save-panels --json results/metrics.json --seed 42

# Early stopping via convergence tolerance
python -m levelset_seg -i data/Te-me_0015.jpg -o result.png --convergence-tol 0.005
```

CLI options:
- `-i, --input`: Path to input image file or directory of images (required).
- `-o, --output`: Path to save output overlay PNG (or directory if `--input` is a directory) (required).
- `--iterations`: Number of PDE iterations (overrides default 85).
- `--convergence-tol`: Stop early when $\max|\phi_{t+1} - \phi_t| < \text{tol}$.
- `--seed`: OpenCV random seed for reproducible K-Means cluster initialization.
- `--save-panels`: Save side-by-side 3-panel visualization (K-Means, ROI, level set).
- `--json [FILE]`: Output execution metrics as JSON (pass a file path, or omit to print to stdout).

### 3. Mask Evaluation Tool (`evaluate.py`)

Compute Dice Similarity Coefficient (DSC) and Intersection over Union (IoU / Jaccard Index) between predicted binary masks and ground-truth annotations:

```bash
# Print metrics to stdout
python evaluate.py --pred path/to/pred_mask.png --gt path/to/gt_mask.png

# Output as JSON
python evaluate.py --pred path/to/pred_mask.png --gt path/to/gt_mask.png --json
```

---

## Running Tests & Linting

Run the test suite with pytest:

```bash
pytest -v
```

The pytest test suite covers:
- Invariants of regularized Heaviside and Dirac delta formulations ($H(0) = 0.5$, $\delta(0) = 1/\pi$, asymptotic limits, monotonicity, symmetry).
- Synthetic disc segmentation integration test asserting contour recovery ($\text{IoU} \ge 0.7$).
- Image scaling, aspect-ratio preservation, CLAHE contrast bounds, and K-Means clustering.
- Topological shape scoring, distance-transform sign convention, and SDF clipping.
- Dice and IoU calculation across identical, disjoint, and partially overlapping masks.
- Full end-to-end headless CLI workflows (single-file, directory batch, `--save-panels`, `--json`).

Run lint checks with Ruff:

```bash
ruff check .
```

---

## How It Works, Panel by Panel

| Panel | Method | Visual Output |
|---|---|---|
| **1. Unsupervised Clusters** | $k$-Means clustering on CLAHE intensity space | Color-coded partition of pixel intensities into 5 discrete clusters. |
| **2. Convex Hull ROI** | Morphological filtering + contour scoring ($\sqrt{A} \cdot S^3 \cdot C^3$) | Selected tumor candidate filled in cyan with its convex hull boundary in green; live display of the calculated solidity metric. |
| **3. Level Set Evolution** | Chan-Vese PDE solved via Euler finite differences | Original MRI slice overlaid with the dynamic $\phi = 0$ contour (neon green), updated in real time as the contour converges. |

---

## Experimental Results

Below are benchmark runtimes and topological metrics recorded across sample images in `data/`:

| Image ID | Dimensions | ROI Solidity | Iterations | Solve Time (s) |
|---|---|---|---|---|
| `Te-me_0015.jpg` | 400 × 335 | 0.9866 | 85 | 1.182s *(first-call JIT compilation)* |
| `Te-me_0020.jpg` | 306 × 306 | 0.9177 | 85 | 0.125s |
| `Te-me_0223.jpg` | 400 × 400 | 0.8901 | 85 | 0.139s |
| `Tr-me_0019.jpg` | 400 × 400 | 0.9176 | 85 | 0.113s |
| `Tr-me_0037.jpg` | 400 × 400 | 0.8953 | 85 | 0.109s |
| `Tr-me_0083.jpg` | 400 × 400 | 0.8951 | 85 | 0.133s |
| `Tr-me_0089.jpg` | 400 × 400 | 0.9190 | 85 | 0.126s |
| `Tr-me_0108.jpg` | 400 × 400 | 0.9827 | 85 | 0.160s |
| `Tr-me_0177.jpg` | 270 × 250 | 0.9723 | 85 | 0.067s |
| `Tr-me_0594.jpg` | 320 × 257 | 0.9528 | 85 | 0.084s |

**Benchmark Provenance**:
- **Platform**: Microsoft Windows 11, x86_64
- **Software**: Python 3.14.3, Numba 0.65.1 (with `cache=True`), NumPy 2.4.4, OpenCV 4.13.0
- **Execution**: Run with `--seed 42` via `python -m levelset_seg -i data/ -o bench/ --json`.
- **Timing Profile**: The first slice includes one-time LLVM JIT compilation overhead (~1.18s); subsequent steady-state iterations execute in ~0.07s–0.16s per image (average steady-state solve time: ~0.11s).
- **Segmentation Accuracy**: Pixel-level evaluation (Dice and IoU) requires ground-truth boundary masks. Quantitative metrics can be generated using [`evaluate.py`](file:///c:/Users/rumki/Desktop/Projects/levelset-brain-tumor-seg/evaluate.py) as annotated masks become available.

---

## Data Source & Attribution

The sample images included in `data/` follow the `Te-me_*` and `Tr-me_*` naming convention matching the [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) curated by Masoud Nickparvar. This aggregated benchmark compiles scans from three primary sources:
1. **Figshare Brain Tumor Dataset** (Jun Cheng et al.)
2. **SARTAJ Dataset**
3. **Br35H Dataset**

- **Class**: Meningioma (`me`) brain MRI slices.
- **License**: See the [Kaggle Dataset Page](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) <!-- TODO: Confirm specific upstream image license terms -->.

---

## Performance Optimizations

1. **Numba JIT Compilation with `parallel=True` and `cache=True`**:
   The PDE update core (`compute_pde_step`) compiles to native machine instructions via LLVM with persistent disk caching. Both the dual spatial reductions (exterior/interior regional averages $c_1, c_2$) and the central finite-difference curvature updates leverage multi-core CPU scheduling through OpenMP/TBB `prange`.
2. **Adaptive Curvature Regularization**:
   The curvature parameter $\mu$ dynamically adapts to image variance ($\mu = 0.25$ for textured/noisy images with variance $\ge 0.05$; $\mu = 0.5$ for homogeneous slices), preventing boundary over-smoothing on delicate margins.
3. **Bounded Aspect-Preserving Scaling**:
   Images exceeding 400px along any dimension are scaled down while maintaining exact aspect ratios, bounding finite-difference operations to $\le 160{,}000$ grid points.
4. **Non-Blocking Background Threading**:
   The GUI decouples numerical simulation from UI rendering via background worker threads (`threading.Thread`), streaming state updates via polling events without freezing window events.

---

## Clinical Disclaimer & Limitations

> [!WARNING]
> **RESEARCH AND EDUCATIONAL USE ONLY**  
> This software is a prototype developed for academic and educational demonstrations. It has **not** been certified, cleared, or validated by the FDA, EMA, or any regulatory medical authority. It must **not** be used for clinical diagnosis, patient treatment planning, or medical decision-making.

### Known Technical Limitations
- **2D Slice Analysis**: Operates on single 2D slices; does not perform 3D anisotropic volumetric reconstruction across z-stacks.
- **Two-phase model**: Standard two-phase Chan-Vese assumes a single background and foreground class; complex multi-component tumors (e.g., heterogeneous glioblastoma with surrounding necrotic core and edema) require multi-phase level-set formulations.
- **Geometric Prior**: The automated ROI selection scoring ($\sqrt{\text{Area}} \cdot \text{Solidity}^3 \cdot \text{Circularity}^3$) specifically favors round, convex regions. While well-suited to meningiomas, it may struggle with irregular, infiltrative, or multi-focal pathologies.
- **Sample Dataset**: The bundled sample images in `data/` consist exclusively of meningioma cases.

---

## References

- T. F. Chan and L. A. Vese, "Active contours without edges," *IEEE Transactions on Image Processing*, vol. 10, no. 2, pp. 266–277, Feb. 2001. doi: [10.1109/83.902291](https://doi.org/10.1109/83.902291).

---

## Citation

If you use this repository in academic or research work, please cite it using the metadata in [CITATION.cff](file:///c:/Users/rumki/Desktop/Projects/levelset-brain-tumor-seg/CITATION.cff):

```bibtex
@software{Paul_levelset_brain_tumor_seg_2026,
  author = {Paul, Maharshi},
  title = {{levelset-brain-tumor-seg: Fully Automatic Level-Set Brain Tumor Segmentation}},
  url = {https://github.com/Maharshi-Paul/levelset-brain-tumor-seg},
  version = {0.1.0},
  year = {2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
