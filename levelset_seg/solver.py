"""Chan-Vese level-set PDE solver (Numba JIT, CPU-parallel).

Implements the classic **"Active Contours Without Edges"** model
(Chan & Vese, 2001).  The tumor boundary is represented implicitly as
the zero level set of a signed distance function φ.  At each Euler
step:

1. **Regional means** ``c₁`` (inside) and ``c₂`` (outside) are
   computed as Heaviside-weighted averages of image intensity.

2. φ is updated according to::

       ∂φ/∂t = δ_ε(φ) · [ μ·κ(φ)  −  λ₁·(I − c₁)²  +  λ₂·(I − c₂)² ]

   where κ is the mean curvature of the level set (computed via
   second-order finite differences), and δ_ε / H_ε are the regularised
   (smoothed) Dirac delta and Heaviside step functions:

       H_ε(φ) = ½ · (1 + (2/π)·arctan(φ/ε))
       δ_ε(φ) = (ε/π) / (ε² + φ²)

   The regularisation width ε prevents numerical instabilities at the
   zero crossing.

The PDE core (``compute_pde_step``) is compiled with ``@jit(nopython=True,
parallel=True)`` so that both the regional-mean reduction and the
curvature update run in parallel across CPU cores via Numba's ``prange``.

All parameters and their defaults are in :class:`~levelset_seg.config.SolverConfig`.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from numba import jit, prange
from numpy.typing import NDArray

from .config import SolverConfig

# ── Regularised Heaviside and Dirac ──────────────────────────────────


@jit(nopython=True, fastmath=True)
def regularized_heaviside(phi: float, epsilon: float = 1.0) -> float:
    r"""Smoothed Heaviside step function.

    .. math::
        H_\varepsilon(\phi) =
            \frac{1}{2}\left(1 + \frac{2}{\pi}\arctan\frac{\phi}{\varepsilon}\right)

    Parameters
    ----------
    phi : float
        Level-set value.
    epsilon : float
        Regularisation width (default 1.0).

    Returns
    -------
    float
        Value in (0, 1).
    """
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / epsilon))


@jit(nopython=True, fastmath=True)
def regularized_dirac(phi: float, epsilon: float = 1.0) -> float:
    r"""Smoothed Dirac delta function.

    .. math::
        \delta_\varepsilon(\phi) =
            \frac{\varepsilon / \pi}{\varepsilon^2 + \phi^2}

    Parameters
    ----------
    phi : float
        Level-set value.
    epsilon : float
        Regularisation width (default 1.0).

    Returns
    -------
    float
        Non-negative value peaking at φ = 0.
    """
    return (epsilon / np.pi) / (epsilon**2 + phi**2)


# ── Numba-parallel PDE kernel ────────────────────────────────────────


@jit(nopython=True, fastmath=True, parallel=True)
def compute_pde_step(
    phi: NDArray[np.float64],
    image: NDArray[np.float64],
    mu: float,
    lambda1: float,
    lambda2: float,
    dt: float,
    epsilon: float,
) -> NDArray[np.float64]:
    """Advance the level-set function φ by one Euler step.

    Implements the two-phase Chan-Vese update:

    1. Compute region means ``c₁``, ``c₂`` via a parallel Heaviside-
       weighted reduction over all pixels.
    2. At each interior pixel compute mean curvature κ from finite
       differences and apply the evolution equation.

    This function is compiled and parallelised by Numba; the first call
    triggers JIT compilation and is slower.

    Parameters
    ----------
    phi : ndarray of float64, shape (H, W)
        Current level-set function.
    image : ndarray of float64, shape (H, W)
        Normalised [0, 1] image.
    mu : float
        Curvature (regularisation) weight.
    lambda1, lambda2 : float
        Inside / outside data-fidelity weights.
    dt : float
        Time step.
    epsilon : float
        Heaviside / Dirac regularisation width.

    Returns
    -------
    ndarray of float64, shape (H, W)
        Updated level-set function.
    """
    rows, cols = phi.shape
    phi_new = np.copy(phi)

    # Phase 1: parallel reduction for regional means
    sum_in = 0.0
    count_in = 0.0
    sum_out = 0.0
    count_out = 0.0

    for i in prange(rows):
        for j in range(cols):
            H = regularized_heaviside(phi[i, j], epsilon)
            sum_in += image[i, j] * H
            count_in += H
            sum_out += image[i, j] * (1.0 - H)
            count_out += (1.0 - H)

    c1 = sum_in / (count_in + 1e-10)
    c2 = sum_out / (count_out + 1e-10)

    # Phase 2: parallel curvature and gradient update
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            phi_x = (phi[i, j + 1] - phi[i, j - 1]) / 2.0
            phi_y = (phi[i + 1, j] - phi[i - 1, j]) / 2.0
            phi_xx = phi[i, j + 1] - 2 * phi[i, j] + phi[i, j - 1]
            phi_yy = phi[i + 1, j] - 2 * phi[i, j] + phi[i - 1, j]
            phi_xy = (
                phi[i + 1, j + 1]
                - phi[i - 1, j + 1]
                - phi[i + 1, j - 1]
                + phi[i - 1, j - 1]
            ) / 4.0

            grad_mag2 = phi_x**2 + phi_y**2 + 1e-10
            kappa = (
                phi_xx * phi_y**2
                - 2 * phi_xy * phi_x * phi_y
                + phi_yy * phi_x**2
            ) / (grad_mag2**1.5)

            delta = regularized_dirac(phi[i, j], epsilon)
            fidelity = -lambda1 * (image[i, j] - c1) ** 2 + lambda2 * (
                image[i, j] - c2
            ) ** 2

            phi_new[i, j] += dt * delta * (mu * kappa + fidelity)

    return phi_new


# ── High-level evolution loop ────────────────────────────────────────


def evolve(
    phi: NDArray[np.float64],
    image: NDArray[np.float64],
    cfg: SolverConfig | None = None,
    on_iteration: Callable[[int, NDArray[np.float64]], None] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Run the full Chan-Vese level-set evolution.

    Parameters
    ----------
    phi : ndarray of float64, shape (H, W)
        Initial level-set function (typically from
        :func:`~levelset_seg.roi.initialise_phi`).
    image : ndarray of float64, shape (H, W)
        Normalised [0, 1] image.
    cfg : SolverConfig, optional
        Configuration.  Uses defaults if *None*.
    on_iteration : callable, optional
        ``on_iteration(i, phi)`` is called after each step (0-indexed).
        Useful for GUI progress updates.

    Returns
    -------
    phi : ndarray of float64, shape (H, W)
        Final level-set function.
    duration : float
        Wall-clock time in seconds.
    """
    if cfg is None:
        cfg = SolverConfig()

    mu = (
        cfg.mu_low_variance
        if np.var(image) < cfg.variance_threshold
        else cfg.mu_high_variance
    )

    start = time.time()
    for i in range(cfg.total_iterations):
        phi_old = phi
        phi = compute_pde_step(
            phi, image, mu, cfg.lambda1, cfg.lambda2, cfg.dt, cfg.epsilon
        )
        phi = np.clip(phi, -cfg.phi_clip, cfg.phi_clip)

        if on_iteration is not None:
            on_iteration(i, phi)

        # Optional convergence-based early stopping
        if cfg.convergence_tol is not None:
            diff = np.max(np.abs(phi - phi_old))
            if diff < cfg.convergence_tol:
                break

    duration = time.time() - start
    return phi, duration
