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

**Fully automatic brain tumor segmentation from MRI — zero manual seeding, zero training data, and zero GPU required.** An unsupervised K-Means and topological ROI search pipeline feeds a Numba CPU-parallel Chan-Vese level-set PDE solver, visualized live in a CustomTkinter desktop dashboard or executed headlessly via CLI.

Built by [Maharshi-Paul](https://github.com/Maharshi-Paul)  
Poster design by [DIPESHCHOUDHARY-CODES](https://github.com/DIPESHCHOUDHARY-CODES)

---

## Table of Contents

- [Problem](#problem)
- [Solution](#solution)
- [On-Device Compute Usage](#on-device-compute-usage)
- [Mathematical Formulation](#mathematical-formulation)
- [Project Architecture](#project-architecture)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Interactive GUI Dashboard](#1-interactive-gui-dashboard)
  - [Headless Command-Line Interface (CLI)](#2-headless-command-line-interface-cli)
- [Running Tests & Linting](#running-tests--linting)
- [How It Works, Panel by Panel](#how-it-works-panel-by-panel)
- [Screenshots & Demo](#screenshots--demo)
- [Experimental Results](#experimental-results)
- [Data Source & Attribution](#data-source--attribution)
- [Performance Optimizations](#performance-optimizations)
- [Clinical Disclaimer & Limitations](#clinical-disclaimer--limitations)
- [License](#license)

---

## Problem

Brain tumor segmentation in magnetic resonance imaging (MRI) is traditionally addressed in two ways:
1. **Manual tracing by a radiologist** — accurate but labour-intensive, subjective, and difficult to scale across clinical workflows.
2. **Deep learning models** — highly capable but dependent on massive pixel-annotated datasets, heavy GPU infrastructure, and susceptible to distribution shift across different scanners, field strengths, and acquisition protocols.

Simpler intensity thresholding breaks down on low-contrast, noisy tissue margins. Classical active-contour (snake) methods require an expert human operator to manually specify a seed point or bounding box inside the lesion before evolution can begin.

---

## Solution

`levelset-brain-tumor-seg` eliminates manual intervention entirely. It provides an end-to-end, deterministic, CPU-only pipeline:

```
[MRI Slice] ──> [CLAHE & Blur] ──> [K-Means (k=5)] ──> [Topological ROI Scoring] ──> [Initial SDF φ₀] ──> [Chan-Vese PDE (Numba CPU)] ──> [Tumor Contour]
```

1. **Contrast Enhancement & Unsupervised Clustering**: Contrast-Limited Adaptive Histogram Equalization (CLAHE) and mild Gaussian filtering enhance tissue contrast, followed by OpenCV K-Means ($k=5$) to partition intensities without supervision.
2. **Topological ROI Discovery**: The brightest tissue clusters are scanned for candidate contours. Each contour is evaluated using a geometric scoring heuristic balancing area, convexity (solidity), and circularity:
   $$\text{Score} = \sqrt{\text{Area}} \cdot \text{Solidity}^3 \cdot \text{Circularity}^3$$
   This automatically isolates meningioma-like circular masses without seed points.
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
| Level-Set PDE Solver | Curvature, Dirac/Heaviside updates, reductions | Numba JIT + `prange` CPU parallelism |

No network telemetry, no external API calls, and no patient data leaves the local machine.

---

## Mathematical Formulation

The tumor boundary $C$ is represented implicitly as the zero level set of a Lipschitz-continuous signed distance function $\phi: \Omega \rightarrow \mathbb{R}$:

$$C = \{ \mathbf{x} \in \Omega \mid \phi(\mathbf{x}) = 0 \}$$

Inside the tumor, $\phi < 0$; outside the tumor, $\phi > 0$.

### Chan-Vese Energy Minimization

The Chan-Vese model (Chan & Vese, 2001) minimizes the energy functional:

$$F(c_1, c_2, \phi) = \mu \int_\Omega \delta(\phi) |\nabla \phi| \, d\mathbf{x} + \lambda_1 \int_\Omega |I - c_1|^2 H(\phi) \, d\mathbf{x} + \lambda_2 \int_\Omega |I - c_2|^2 (1 - H(\phi)) \, d\mathbf{x}$$

where:
- $I(\mathbf{x}) \in [0, 1]$ is the normalized image intensity.
- $c_1$ and $c_2$ are the Heaviside-weighted average intensities inside ($\phi < 0$) and outside ($\phi > 0$) the contour:
  $$c_1 = \frac{\int_\Omega I \cdot (1 - H(\phi)) \, d\mathbf{x}}{\int_\Omega (1 - H(\phi)) \, d\mathbf{x}}, \quad c_2 = \frac{\int_\Omega I \cdot H(\phi) \, d\mathbf{x}}{\int_\Omega H(\phi) \, d\mathbf{x}}$$
- $\mu$ controls boundary smoothness via curvature regularisation.
- $\lambda_1, \lambda_2$ scale internal and external data fidelity.

### Euler-Lagrange Evolution Equation

Minimizing $F$ with respect to $\phi$ produces the parabolic PDE:

$$\frac{\partial \phi}{\partial t} = \delta_\varepsilon(\phi) \left[ \mu \cdot \kappa(\phi) - \lambda_1 (I - c_1)^2 + \lambda_2 (I - c_2)^2 \right]$$

where $\kappa(\phi) = \text{div}\left(\frac{\nabla \phi}{|\nabla \phi|}\right)$ is the mean curvature computed using central finite differences:

$$\kappa = \frac{\phi_{xx} \phi_y^2 - 2 \phi_x \phi_y \phi_{xy} + \phi_{yy} \phi_x^2}{(\phi_x^2 + \phi_y^2 + \eta)^{3/2}}$$

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
| `lambda1`, `lambda2` | `1.0`, `1.0` | Interior and exterior region-fitting weights |
| `mu` | Adaptive (`0.25` or `0.5`) | Curvature regularization weight ($0.25$ if $\text{Var}(I) \ge 0.05$, else $0.5$) |
| `epsilon` | `1.0` | Heaviside / Dirac regularization width |
| `max_dim` | `400` | Maximum spatial dimension for image scaling |
| `kmeans_k` | `5` | Number of clusters for tissue partitioning |
| `phi_clip` | `5.0` | Signed distance clamp interval $[-5.0, +5.0]$ |
| `convergence_tol` | `None` | Optional early stopping tolerance $\max|\phi_{t+1} - \phi_t| < \text{tol}$ |

---

## Project Architecture

```
levelset-brain-tumor-seg/
├── levelset_seg/                  # Core package
│   ├── __init__.py                # Package export & version metadata
│   ├── __main__.py                # CLI entry point (python -m levelset_seg)
│   ├── cli.py                     # Headless command-line interface
│   ├── config.py                  # SolverConfig dataclass (single source of truth)
│   ├── gui.py                     # CustomTkinter + Matplotlib dashboard
│   ├── preprocessing.py          # CLAHE, Gaussian blur, scaling, K-Means
│   ├── roi.py                     # Topological contour scoring & SDF creation
│   └── solver.py                  # Numba-parallelized Chan-Vese PDE solver
├── tests/                         # Pytest test suite
│   ├── __init__.py
│   ├── test_cli.py                # Headless CLI integration tests
│   ├── test_preprocessing.py      # CLAHE & K-Means unit tests
│   ├── test_roi.py                # Shape scoring & SDF clamp tests
│   └── test_solver.py             # Math invariants (Heaviside/Dirac) & synthetic IoU
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI matrix (Python 3.10–3.14)
├── assets/                        # Visual banners & UI assets
│   └── levelset_banner.png
├── data/                          # Sample axial MRI brain slices (meningioma)
├── main.py                        # Thin application launcher
├── pyproject.toml                 # Ruff linting & pytest configuration
├── requirements.txt               # Pinned core production dependencies
├── requirements-dev.txt           # Testing & linting tools
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

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To install development and testing tools:
```bash
pip install -r requirements-dev.txt
```

> **Note on Numba Compilation**: When running for the first time, Numba JIT-compiles `compute_pde_step` and mathematical kernels. Subsequent steps execute at native C-level speed.

---

## Usage

### 1. Interactive GUI Dashboard

Launch the desktop application:

```bash
python main.py
```

1. Click **Upload Batch & Segment** and choose one or more axial MRI images from `data/` or your own files (`.png`, `.jpg`, `.jpeg`, `.tif`, `.bmp`).
2. Watch the three-panel display:
   - Panel 1 shows unsupervised intensity clustering.
   - Panel 2 isolates the candidate tumor contour and highlights its convex hull with live solidity metrics.
   - Panel 3 visualizes the real-time PDE contour evolution overlaid on the MRI slice.
3. Use the **Prev** and **Next** navigation buttons to step through segmented images in the batch.

### 2. Headless Command-Line Interface (CLI)

For batch processing, automated scripting, or headless servers (without Tkinter/X11):

```bash
# Basic headless segmentation
python -m levelset_seg --input data/Te-me_0015.jpg --output output_overlay.png

# Custom iterations and reproducible RNG seed
python -m levelset_seg -i data/Tr-me_0019.jpg -o tumor_seg.png --iterations 60 --seed 42

# Early stopping via convergence tolerance
python -m levelset_seg -i data/Te-me_0083.jpg -o result.png --convergence-tol 0.005
```

CLI options:
- `-i, --input`: Path to input MRI slice (required).
- `-o, --output`: Path to save output contour overlay PNG (required).
- `--iterations`: Number of PDE iterations (overrides default 85).
- `--convergence-tol`: Stop early when $\max|\phi_{t+1} - \phi_t| < \text{tol}$.
- `--seed`: OpenCV random seed for deterministic K-Means cluster initialization.

---

## Running Tests & Linting

Run the complete test suite:

```bash
pytest -v
```

All 27 unit and integration tests execute headlessly on synthetic images and mathematical functions:
- Unit verification of smoothed Heaviside and Dirac properties ($H(0) = 0.5$, $\delta(0) = 1/\pi$, asymptotic limits).
- Synthetic bright-disc segmentation integration tests asserting contour overlap ($\text{IoU} \ge 0.7$).
- Image scaling, CLAHE contrast bounds, and K-Means shape verification.
- End-to-end CLI workflow tests.

Run lint checks with Ruff:

```bash
ruff check .
```

---

## How It Works, Panel by Panel

| Panel | Method | Visual Output |
|---|---|---|
| **1. Unsupervised Clusters** | $k$-Means clustering on CLAHE intensity space | Color-coded cluster partition representing distinct brain tissues (CSF, grey matter, white matter, pathology). |
| **2. Convex Hull ROI** | Morphological filtering + contour scoring ($\sqrt{A} \cdot S^3 \cdot C^3$) | Selected tumor candidate filled in cyan with its convex hull boundary in green; live display of the calculated solidity metric. |
| **3. Level Set Evolution** | Chan-Vese PDE solved via Euler finite differences | Original MRI slice overlaid with the dynamic $\phi = 0$ contour (neon green), updated in real time as the contour converges. |

---

## Screenshots & Demo

<!-- TODO: Capture real screenshots and animations of the running CustomTkinter dashboard using images from data/ -->

| Full Application Dashboard | Dynamic Contour Evolution |
|:---:|:---:|
| ![Dashboard Screenshot](assets/dashboard.png) <br> *(TODO: Replace with dashboard.png screenshot)* | ![Segmentation Demo GIF](assets/demo.gif) <br> *(TODO: Replace with demo.gif animation)* |

---

## Experimental Results

<!-- TODO: Once ground-truth annotations are available for the data/ set, calculate and document exact Dice and IoU metrics here -->

| Image ID | Tumor Type | Resolution | Slice Solidity | Dice Score | IoU (Jaccard) | Iterations | Solve Time (s) |
|---|---|---|---|---|---|---|---|
| `Te-me_0015` | Meningioma | 400 × 335 | 0.987 | *TODO* | *TODO* | 85 | ~2.6s |
| `Te-me_0041` | Meningioma | 400 × 400 | 0.965 | *TODO* | *TODO* | 85 | ~2.5s |
| `Tr-me_0019` | Meningioma | 400 × 382 | 0.978 | *TODO* | *TODO* | 85 | ~2.4s |
| `Tr-me_0060` | Meningioma | 400 × 356 | 0.952 | *TODO* | *TODO* | 85 | ~2.5s |

*Note: Dice and IoU evaluation require pixel-level ground-truth masks. The table above reflects sample execution times and detected solidity scores on an Intel/AMD multicore CPU.*

---

## Data Source & Attribution

The sample images included in `data/` follow the `Te-me_*` and `Tr-me_*` naming convention matching the [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) curated by Masoud Nickparvar. This aggregated benchmark compiles scans from three primary sources:
1. **Figshare Brain Tumor Dataset** (Jun Cheng et al.)
2. **SARTAJ Dataset**
3. **Br35H Dataset**

- **Class**: Meningioma (`me`) axial T1-weighted contrast-enhanced slices.
- **License**: See the [Kaggle Dataset Page](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) <!-- TODO: Confirm specific upstream image license terms -->.

---

## Performance Optimizations

1. **Numba JIT Compilation with `parallel=True`**:
   The PDE update core (`compute_pde_step`) compiles down to native machine code at runtime via LLVM. Both the dual spatial reductions (interior/exterior regional averages $c_1, c_2$) and the central finite-difference curvature updates leverage multi-core CPU scheduling through OpenMP/TBB `prange`.
2. **Adaptive Curvature Regularization**:
   The curvature parameter $\mu$ dynamically adapts to image variance ($\mu = 0.25$ for textured/noisy images with variance $\ge 0.05$; $\mu = 0.5$ for homogeneous slices), preventing boundary over-smoothing on delicate margins.
3. **Bounded Aspect-Preserving Scaling**:
   Images exceeding 400px along any dimension are scaled down while maintaining exact aspect ratios, bounding finite-difference operations to $\le 160{,}000$ grid points without sacrificing clinical detail.
4. **Non-Blocking Background Threading**:
   The GUI decouples numerical simulation from UI rendering via background worker threads (`threading.Thread`), streaming state updates via polling events without freezing window events.

---

## Clinical Disclaimer & Limitations

> [!WARNING]
> **RESEARCH AND EDUCATIONAL USE ONLY**  
> This software is a prototype developed for academic and educational demonstrations. It has **not** been certified, cleared, or validated by the FDA, EMA, or any regulatory medical authority. It must **not** be used for clinical diagnosis, patient treatment planning, or medical decision-making.

### Known Technical Limitations
- **2D Slice Analysis**: Operates on single 2D slices; does not perform 3D anisotropic volumetric reconstruction across z-stacks.
- **Intensity Prior**: The automated ROI search assumes tumor tissue falls within the top three brightest intensity clusters in contrast-enhanced T1 MRI. Hypointense or non-enhancing necrotic cores may require adjusted search clusters.
- **Topology Changes**: Standard two-phase Chan-Vese assumes a single background and foreground class; complex multi-component tumors (e.g., heterogeneous glioblastoma with surrounding edema) require multi-phase level-set extensions.

---

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
