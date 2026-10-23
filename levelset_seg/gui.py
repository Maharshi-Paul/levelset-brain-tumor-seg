"""CustomTkinter + Matplotlib live dashboard.

Provides :class:`BrainTumorMasterApp`, a desktop GUI that lets the user
upload one or more MRI images, runs the full segmentation pipeline in a
background thread, and renders the three-panel view (K-Means clusters,
topological ROI, level-set evolution) in real time.

All numerical behaviour is delegated to :mod:`levelset_seg.preprocessing`,
:mod:`levelset_seg.roi`, and :mod:`levelset_seg.solver`.
"""

from __future__ import annotations

import threading

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

from .config import SolverConfig
from .preprocessing import preprocess
from .roi import find_best_roi, initialise_phi
from .solver import evolve, compute_pde_step

# Set modern appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class BrainTumorMasterApp(ctk.CTk):
    """Main dashboard window for the segmentation pipeline."""

    def __init__(self, cfg: SolverConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SolverConfig()
        self.title("Medical Image PDE Segmentation Dashboard")
        self.geometry("1500x800")

        # State Variables
        self.image_paths: list[str] = []
        self.current_idx: int = 0
        self.image: np.ndarray | None = None
        self.phi: np.ndarray | None = None
        self.kmeans_vis: np.ndarray | None = None
        self.topology_vis: np.ndarray | None = None
        self.best_solidity: float = 0.0

        # Threading controls
        self.is_simulating: bool = False
        self.current_iteration: int = 0
        self.sim_duration: float = 0.0

        # Matplotlib Dark Theme Styling
        plt.style.use("dark_background")

        self.build_ui()

    def build_ui(self) -> None:
        """Construct all GUI widgets."""
        # Configure grid layout (1 row, 2 columns)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT SIDEBAR ---
        self.sidebar_frame = ctk.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Neural PDE Engine",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))

        self.btn_upload = ctk.CTkButton(
            self.sidebar_frame,
            text="Upload Batch & Segment",
            command=self.upload_images,
            height=40,
        )
        self.btn_upload.grid(row=1, column=0, padx=20, pady=20)

        # Navigation Frame
        self.nav_frame = ctk.CTkFrame(
            self.sidebar_frame, fg_color="transparent"
        )
        self.nav_frame.grid(row=2, column=0, padx=20, pady=10)

        self.btn_prev = ctk.CTkButton(
            self.nav_frame,
            text="<< Prev",
            width=80,
            command=self.prev_image,
            state="disabled",
        )
        self.btn_prev.grid(row=0, column=0, padx=5)

        self.lbl_counter = ctk.CTkLabel(
            self.nav_frame, text="0 / 0", font=ctk.CTkFont(size=14)
        )
        self.lbl_counter.grid(row=0, column=1, padx=15)

        self.btn_next = ctk.CTkButton(
            self.nav_frame,
            text="Next >>",
            width=80,
            command=self.next_image,
            state="disabled",
        )
        self.btn_next.grid(row=0, column=2, padx=5)

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.sidebar_frame)
        self.progress_bar.grid(row=3, column=0, padx=20, pady=(20, 5))
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(
            self.sidebar_frame, text="System Idle", text_color="gray"
        )
        self.lbl_status.grid(row=4, column=0, padx=20, pady=0)

        # Console Log
        self.console = ctk.CTkTextbox(
            self.sidebar_frame,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.console.grid(row=5, column=0, padx=20, pady=20, sticky="nsew")
        self.log("JIIT Core Initialized. Awaiting Data...")

        # --- RIGHT DISPLAY AREA ---
        self.display_frame = ctk.CTkFrame(self, corner_radius=10)
        self.display_frame.grid(
            row=0, column=1, padx=20, pady=20, sticky="nsew"
        )

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5))
        self.fig.patch.set_facecolor("#2b2b2b")
        for ax in self.axes:
            ax.axis("off")
            ax.set_facecolor("#2b2b2b")

        self.axes[0].set_title(
            r"1. K-Means Clustering", fontsize=12, color="white"
        )
        self.axes[1].set_title(
            r"2. Topological Search", fontsize=12, color="white"
        )
        self.axes[2].set_title(
            r"3. Level Set $\phi$ Evolution", fontsize=12, color="white"
        )
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=10, pady=10
        )
        self.canvas.draw()

    def log(self, message: str) -> None:
        """Append a line to the console log."""
        self.console.insert("end", message + "\n")
        self.console.see("end")

    def upload_images(self) -> None:
        """Open file dialog and start processing the first image."""
        files = filedialog.askopenfilenames(
            title="Select Brain MRI Images",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.tif *.bmp")
            ],
        )
        if not files:
            return
        self.image_paths = list(files)
        self.current_idx = 0
        self.log(
            f"\nBatch Loaded: {len(self.image_paths)} image(s) via I/O stream."
        )
        self.update_navigator()
        self.process_current_image()

    def update_navigator(self) -> None:
        """Update prev/next buttons and counter label."""
        total = len(self.image_paths)
        self.lbl_counter.configure(
            text=f"{self.current_idx + 1} / {total}"
        )
        self.btn_prev.configure(
            state=(
                "normal"
                if self.current_idx > 0 and not self.is_simulating
                else "disabled"
            )
        )
        self.btn_next.configure(
            state=(
                "normal"
                if self.current_idx < total - 1 and not self.is_simulating
                else "disabled"
            )
        )

    def prev_image(self) -> None:
        """Navigate to the previous image in the batch."""
        if self.current_idx > 0 and not self.is_simulating:
            self.current_idx -= 1
            self.update_navigator()
            self.process_current_image()

    def next_image(self) -> None:
        """Navigate to the next image in the batch."""
        if (
            self.current_idx < len(self.image_paths) - 1
            and not self.is_simulating
        ):
            self.current_idx += 1
            self.update_navigator()
            self.process_current_image()

    def process_current_image(self) -> None:
        """Run the full preprocessing + ROI pipeline, then start the solver."""
        self.btn_upload.configure(state="disabled")
        self.btn_prev.configure(state="disabled")
        self.btn_next.configure(state="disabled")
        self.progress_bar.set(0)

        file_path = self.image_paths[self.current_idx]
        self.log(f"\n[+] Analyzing Tensor: {file_path.split('/')[-1]}")

        # Preprocessing
        self.log("[-] Executing Unsupervised Multi-Phase Clustering...")
        image, labels, centers, self.kmeans_vis, img_raw = preprocess(
            file_path, self.cfg
        )
        self.image = image

        # ROI search
        self.log("[-] Mapping Geometry and Convexity...")
        mask_clean, self.topology_vis, self.best_solidity = find_best_roi(
            labels, centers, img_raw.shape, self.cfg
        )

        if self.best_solidity > 0:
            self.log(
                f"[✓] Topological ROI locked. "
                f"Solidity: {self.best_solidity:.3f}"
            )
        else:
            self.log(
                "[!] Warning: Mathematical topology failed. "
                "Falling back to global mass."
            )

        # Initialise φ
        self.phi = initialise_phi(mask_clean, self.cfg.phi_clip)

        self.update_canvas(
            pde_title="Initialization Sequence ($T=0$)"
        )

        # Start async simulation
        self.is_simulating = True
        self.current_iteration = 0
        self.lbl_status.configure(
            text="Solving Euler-Lagrange...", text_color="#00ff00"
        )
        threading.Thread(
            target=self._pde_simulation_thread, daemon=True
        ).start()
        self._poll_simulation_updates()

    def _pde_simulation_thread(self) -> None:
        """Run the Numba JIT PDE on a background thread."""
        mu = (
            self.cfg.mu_low_variance
            if np.var(self.image) < self.cfg.variance_threshold
            else self.cfg.mu_high_variance
        )

        import time

        start_time = time.time()
        for i in range(self.cfg.total_iterations):
            self.phi = compute_pde_step(
                self.phi,
                self.image,
                mu,
                self.cfg.lambda1,
                self.cfg.lambda2,
                self.cfg.dt,
                self.cfg.epsilon,
            )
            self.phi = np.clip(
                self.phi, -self.cfg.phi_clip, self.cfg.phi_clip
            )
            self.current_iteration = i + 1

        self.sim_duration = time.time() - start_time
        self.is_simulating = False

    def _poll_simulation_updates(self) -> None:
        """Polled by Tkinter to update GUI safely from the main thread."""
        if self.is_simulating:
            progress = (
                self.current_iteration / self.cfg.total_iterations
            )
            self.progress_bar.set(progress)

            if self.current_iteration % self.cfg.gui_redraw_every == 0:
                self.update_canvas(
                    pde_title=(
                        rf"Evolution: T={self.current_iteration}"
                        rf"/{self.cfg.total_iterations}"
                    )
                )

            self.after(self.cfg.gui_poll_ms, self._poll_simulation_updates)
        else:
            # Simulation finished
            self.progress_bar.set(1.0)
            self.lbl_status.configure(
                text="Segmentation Complete", text_color="white"
            )
            self.log(
                f"[✓] Multi-threaded convergence achieved "
                f"in {self.sim_duration:.3f}s"
            )
            self.update_canvas(
                pde_title=r"Final Mathematical Boundary ($\phi=0$)"
            )
            self.btn_upload.configure(state="normal")
            self.update_navigator()

    def update_canvas(self, pde_title: str) -> None:
        """Re-render the Matplotlib canvas with current state."""
        for ax in self.axes:
            ax.clear()

        self.axes[0].imshow(self.kmeans_vis, cmap="magma")
        self.axes[0].set_title(
            "1. Unsupervised Clusters", fontsize=11, color="white"
        )

        self.axes[1].imshow(self.topology_vis)
        self.axes[1].set_title(
            rf"2. Convex Hull ROI (S={self.best_solidity:.2f})",
            fontsize=11,
            color="white",
        )

        self.axes[2].imshow(self.image, cmap="gray")
        self.axes[2].contour(
            self.phi, [0], colors="#39ff14", linewidths=2.5
        )
        self.axes[2].set_title(pde_title, fontsize=11, color="white")

        for ax in self.axes:
            ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()
