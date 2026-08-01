import numpy as np
import cv2
import time
import threading
import customtkinter as ctk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from numba import jit, prange

# Set modern appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==========================================
# 1. High-Performance PDE Core (Numba JIT + Parallel)
# ==========================================
@jit(nopython=True, fastmath=True)
def regularized_heaviside(phi, epsilon=1.0):
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / epsilon))

@jit(nopython=True, fastmath=True)
def regularized_dirac(phi, epsilon=1.0):
    return (epsilon / np.pi) / (epsilon**2 + phi**2)

# OVERRIDE: parallel=True forces CPU to use all available cores
@jit(nopython=True, fastmath=True, parallel=True)
def compute_pde_step(phi, image, mu, lambda1, lambda2, dt, epsilon):
    rows, cols = phi.shape
    phi_new = np.copy(phi)
    
    # Phase 1: Parallel reduction for regional means
    sum_in = 0.0
    count_in = 0.0
    sum_out = 0.0
    count_out = 0.0
    
    # prange enables multi-threaded execution
    for i in prange(rows):
        for j in range(cols):
            H = regularized_heaviside(phi[i, j], epsilon)
            sum_in += image[i, j] * H
            count_in += H
            sum_out += image[i, j] * (1.0 - H)
            count_out += (1.0 - H)
            
    c1 = sum_in / (count_in + 1e-10)
    c2 = sum_out / (count_out + 1e-10)

    # Phase 2: Parallel curvature and gradient computation
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            phi_x = (phi[i, j+1] - phi[i, j-1]) / 2.0
            phi_y = (phi[i+1, j] - phi[i-1, j]) / 2.0
            phi_xx = phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]
            phi_yy = phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]
            phi_xy = (phi[i+1, j+1] - phi[i-1, j+1] - phi[i+1, j-1] + phi[i-1, j-1]) / 4.0
            
            grad_mag2 = phi_x**2 + phi_y**2 + 1e-10
            kappa = (phi_xx * phi_y**2 - 2 * phi_xy * phi_x * phi_y + phi_yy * phi_x**2) / (grad_mag2**1.5)
            
            delta = regularized_dirac(phi[i, j], epsilon)
            fidelity = -lambda1 * (image[i, j] - c1)**2 + lambda2 * (image[i, j] - c2)**2
            
            phi_new[i, j] += dt * delta * (mu * kappa + fidelity)
            
    return phi_new

# ==========================================
# 2. Master GUI Application (CustomTkinter + Async)
# ==========================================
class BrainTumorMasterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Medical Image PDE Segmentation Dashboard")
        self.geometry("1500x800")
        
        # State Variables
        self.image_paths = []
        self.current_idx = 0
        self.image = None
        self.phi = None
        self.kmeans_vis = None
        self.topology_vis = None
        self.best_solidity = 0.0
        
        # Threading controls
        self.is_simulating = False
        self.current_iteration = 0
        self.total_iterations = 85
        
        # Matplotlib Dark Theme Styling
        plt.style.use('dark_background')
        
        self.build_ui()
        
    def build_ui(self):
        # Configure grid layout (1 row, 2 columns)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- LEFT SIDEBAR ---
        self.sidebar_frame = ctk.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Neural PDE Engine", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        self.btn_upload = ctk.CTkButton(self.sidebar_frame, text="Upload Batch & Segment", command=self.upload_images, height=40)
        self.btn_upload.grid(row=1, column=0, padx=20, pady=20)
        
        # Navigation Frame
        self.nav_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.nav_frame.grid(row=2, column=0, padx=20, pady=10)
        
        self.btn_prev = ctk.CTkButton(self.nav_frame, text="<< Prev", width=80, command=self.prev_image, state="disabled")
        self.btn_prev.grid(row=0, column=0, padx=5)
        
        self.lbl_counter = ctk.CTkLabel(self.nav_frame, text="0 / 0", font=ctk.CTkFont(size=14))
        self.lbl_counter.grid(row=0, column=1, padx=15)
        
        self.btn_next = ctk.CTkButton(self.nav_frame, text="Next >>", width=80, command=self.next_image, state="disabled")
        self.btn_next.grid(row=0, column=2, padx=5)
        
        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.sidebar_frame)
        self.progress_bar.grid(row=3, column=0, padx=20, pady=(20, 5))
        self.progress_bar.set(0)
        
        self.lbl_status = ctk.CTkLabel(self.sidebar_frame, text="System Idle", text_color="gray")
        self.lbl_status.grid(row=4, column=0, padx=20, pady=0)
        
        # Console Log
        self.console = ctk.CTkTextbox(self.sidebar_frame, font=ctk.CTkFont(family="Consolas", size=12))
        self.console.grid(row=5, column=0, padx=20, pady=20, sticky="nsew")
        self.log("JIIT Core Initialized. Awaiting Data...")
        
        # --- RIGHT DISPLAY AREA ---
        self.display_frame = ctk.CTkFrame(self, corner_radius=10)
        self.display_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5))
        self.fig.patch.set_facecolor('#2b2b2b')
        for ax in self.axes: 
            ax.axis('off')
            ax.set_facecolor('#2b2b2b')
            
        self.axes[0].set_title(r"1. K-Means Clustering", fontsize=12, color="white")
        self.axes[1].set_title(r"2. Topological Search", fontsize=12, color="white")
        self.axes[2].set_title(r"3. Level Set $\phi$ Evolution", fontsize=12, color="white")
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.draw()

    def log(self, message):
        self.console.insert("end", message + "\n")
        self.console.see("end")

    def upload_images(self):
        files = filedialog.askopenfilenames(title="Select Brain MRI Images", 
                                            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.bmp")])
        if not files: return
        self.image_paths = list(files)
        self.current_idx = 0
        self.log(f"\nBatch Loaded: {len(self.image_paths)} image(s) via I/O stream.")
        self.update_navigator()
        self.process_current_image()

    def update_navigator(self):
        total = len(self.image_paths)
        self.lbl_counter.configure(text=f"{self.current_idx + 1} / {total}")
        self.btn_prev.configure(state="normal" if self.current_idx > 0 and not self.is_simulating else "disabled")
        self.btn_next.configure(state="normal" if self.current_idx < total - 1 and not self.is_simulating else "disabled")

    def prev_image(self):
        if self.current_idx > 0 and not self.is_simulating:
            self.current_idx -= 1
            self.update_navigator()
            self.process_current_image()

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1 and not self.is_simulating:
            self.current_idx += 1
            self.update_navigator()
            self.process_current_image()

    def process_current_image(self):
        """Mathematical Pipeline execution before Threading"""
        self.btn_upload.configure(state="disabled")
        self.btn_prev.configure(state="disabled")
        self.btn_next.configure(state="disabled")
        self.progress_bar.set(0)
        
        file_path = self.image_paths[self.current_idx]
        self.log(f"\n[+] Analyzing Tensor: {file_path.split('/')[-1]}")
        img_raw = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # Scale for optimal PDE Performance
        max_dim = 400
        h, w = img_raw.shape
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img_raw = cv2.resize(img_raw, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # Contrast Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_raw)
        img_smooth = cv2.GaussianBlur(img_clahe, (5, 5), 0)
        self.image = img_smooth.astype(np.float64) / 255.0

        # Math Step 1: K-Means Statistical Clustering
        self.log("[-] Executing Unsupervised Multi-Phase Clustering...")
        pixel_values = np.float32(img_smooth.reshape((-1, 1)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        self.kmeans_vis = (labels.flatten() * 51).reshape(img_raw.shape).astype(np.uint8)
        
        # Math Step 2: Multi-Phase Universal Topological Search
        self.log("[-] Mapping Geometry and Convexity...")
        sorted_centers_idx = np.argsort(centers.flatten())
        search_clusters = sorted_centers_idx[-3:]
        
        best_contour = None
        max_score = -1
        self.best_solidity = 0.0
        self.topology_vis = cv2.cvtColor(np.zeros_like(img_raw), cv2.COLOR_GRAY2RGB)
        mask_clean = np.zeros_like(img_raw)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        for cluster_idx in search_clusters:
            mask = (labels.flatten() == cluster_idx).reshape(img_raw.shape).astype(np.uint8) * 255
            mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100: continue
                
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                solidity = area / hull_area
                
                score = np.sqrt(area) * (solidity ** 3) * (circularity ** 3)
                if score > max_score:
                    max_score = score
                    best_contour = cnt
                    self.best_solidity = solidity

        if best_contour is not None:
            cv2.drawContours(mask_clean, [best_contour], -1, 255, thickness=cv2.FILLED)
            # Draw highly visible neon highlights
            cv2.drawContours(self.topology_vis, [best_contour], -1, (0, 255, 255), thickness=cv2.FILLED)
            cv2.drawContours(self.topology_vis, [cv2.convexHull(best_contour)], -1, (0, 255, 0), 2)
            self.log(f"[✓] Topological ROI locked. Solidity: {self.best_solidity:.3f}")
        else:
            self.log("[!] Warning: Mathematical topology failed. Falling back to global mass.")
            mask_clean = (labels.flatten() == sorted_centers_idx[-1]).reshape(img_raw.shape).astype(np.uint8) * 255

        # Initialize Signed Distance Function (SDF)
        dist_inside = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 3)
        dist_outside = cv2.distanceTransform(255 - mask_clean, cv2.DIST_L2, 3)
        self.phi = np.clip(dist_outside - dist_inside, -5.0, 5.0)
        
        self.update_canvas(pde_title="Initialization Sequence ($T=0$)")
        
        # Start Async Simulation
        self.is_simulating = True
        self.current_iteration = 0
        self.lbl_status.configure(text="Solving Euler-Lagrange...", text_color="#00ff00")
        threading.Thread(target=self.pde_simulation_thread, daemon=True).start()
        self.poll_simulation_updates()

    def pde_simulation_thread(self):
        """Runs the Numba JIT PDE on a separate CPU thread."""
        dt = 0.5 
        lambda1, lambda2 = 1.0, 1.0
        adaptive_mu = 0.5 if np.var(self.image) < 0.05 else 0.25
        
        start_time = time.time()
        for i in range(self.total_iterations):
            self.phi = compute_pde_step(self.phi, self.image, adaptive_mu, lambda1, lambda2, dt, 1.0)
            self.phi = np.clip(self.phi, -5.0, 5.0) 
            self.current_iteration = i + 1
            
        self.sim_duration = time.time() - start_time
        self.is_simulating = False

    def poll_simulation_updates(self):
        """Polled by Tkinter to update GUI safely without thread clashes."""
        if self.is_simulating:
            # Update Progress Bar
            progress = self.current_iteration / self.total_iterations
            self.progress_bar.set(progress)
            
            # Redraw visually every 10 frames
            if self.current_iteration % 10 == 0:
                self.update_canvas(pde_title=rf"Evolution: T={self.current_iteration}/{self.total_iterations}")
            
            # Re-poll in 50ms
            self.after(50, self.poll_simulation_updates)
        else:
            # Simulation Finished
            self.progress_bar.set(1.0)
            self.lbl_status.configure(text="Segmentation Complete", text_color="white")
            self.log(f"[✓] Multi-threaded convergence achieved in {self.sim_duration:.3f}s")
            self.update_canvas(pde_title=r"Final Mathematical Boundary ($\phi=0$)")
            self.btn_upload.configure(state="normal")
            self.update_navigator()

    def update_canvas(self, pde_title):
        """Renders the Matplotlib canvas with current data state."""
        for ax in self.axes: ax.clear()
        
        self.axes[0].imshow(self.kmeans_vis, cmap='magma')
        self.axes[0].set_title("1. Unsupervised Clusters", fontsize=11, color="white")
        
        self.axes[1].imshow(self.topology_vis)
        self.axes[1].set_title(rf"2. Convex Hull ROI (S={self.best_solidity:.2f})", fontsize=11, color="white")
        
        self.axes[2].imshow(self.image, cmap='gray')
        # Neon green level set contour
        self.axes[2].contour(self.phi, [0], colors='#39ff14', linewidths=2.5)
        self.axes[2].set_title(pde_title, fontsize=11, color="white")
        
        for ax in self.axes: ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = BrainTumorMasterApp()
    app.mainloop()