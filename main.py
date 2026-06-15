import os
import sys

# ── pip install numpy ──────────────────────────────────────────────────────
import numpy as np         

# ── pip install opencv-python ─────────────────────────────────────────────
import cv2                

# ── pip install matplotlib ────────────────────────────────────────────────
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# ── pip install pillow ────────────────────────────────────────────────────
from PIL import Image        
# ── pip install trimesh ───────────────────────────────────────────────────
import trimesh            



INPUT_IMAGE   = "input.png"  

OUTPUT_MESH   = "terrain.obj" 
OUTPUT_NORMAL = "normal_map.png" 

MESH_WIDTH    = 128           # Mesh grid columns (higher = more detail, slower)
MESH_HEIGHT   = 128           # Mesh grid rows
HEIGHT_SCALE  = 10.0          # Max elevation in 3D units (increase for dramatic terrain)
SMOOTH_SIGMA  = 1.5           # Gaussian blur before height extraction (0 = off)
                               # Higher values = smoother, rounder terrain




def load_image(path: str) -> np.ndarray:
    
    if os.path.exists(path):
        
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"OpenCV could not read '{path}'. Check the file format.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        print(f"[✓] Loaded image: {path}  ({img.shape[1]}×{img.shape[0]} px)")
    else:
        # ── Demo image: concentric hills + Perlin-like noise ──────────────
        print(f"[!] '{path}' not found — generating a demo terrain image.")
        size = 256
        y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]

        # Radial hill in the center
        hill   = np.exp(-(x**2 + y**2) / 0.3)

        # Two offset secondary bumps
        bump1  = 0.5 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
        bump2  = 0.4 * np.exp(-((x+0.4)**2 + (y-0.3)**2) / 0.08)

        # Low-frequency pseudo-noise via stacked sines
        noise  = 0.1 * (np.sin(x * 8) * np.cos(y * 6) +
                        np.sin(x * 13 + 1) * np.sin(y * 11 + 2))

        raw    = hill + bump1 + bump2 + noise
        raw    = (raw - raw.min()) / (raw.max() - raw.min())   # normalise → [0,1]
        img    = np.stack([raw, raw, raw], axis=-1).astype(np.float32)
        print(f"[✓] Demo image created: {size}×{size} px")

    return img



def image_to_heightmap(img_rgb: np.ndarray,
                       target_w: int,
                       target_h: int,
                       sigma: float = 1.5) -> np.ndarray:
    """
    Mathematics:
        Luminance = 0.299·R + 0.587·G + 0.114·B   (ITU-R BT.601 standard)
        This weights green most heavily because human eyes are most sensitive
        to green wavelengths.
    """

    
    # ── Luminance conversion ──────────────────────────────────────────────

    gray = (0.299 * img_rgb[:, :, 0] +
            0.587 * img_rgb[:, :, 1] +
            0.114 * img_rgb[:, :, 2])

    # ── Optional Gaussian smooth ──────────────────────────────────────────

    if sigma > 0:
        ksize = int(6 * sigma) | 1          # kernel must be odd
        gray  = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)

    # ── Resize to mesh resolution ─────────────────────────────────────────

    hmap = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # ── Re-normalise after blur/resize ────────────────────────────────────
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

    print(f"[✓] Height map computed: {target_w}×{target_h}  "
          f"(min={hmap.min():.3f}, max={hmap.max():.3f})")
    return hmap.astype(np.float32)


# =============================================================================
# STEP 3 — COMPUTE NORMAL MAP FROM HEIGHT MAP
# =============================================================================

def heightmap_to_normalmap(hmap: np.ndarray,
                           height_scale: float = 1.0) -> np.ndarray:
    """
    

    MATHEMATICS EXPLAINED:
    ─────────────────────
    A normal map stores, per pixel, the surface normal vector N = (Nx, Ny, Nz).
    We compute it by treating the height map h(x,y) as a scalar function and
    finding its spatial derivatives:

        ∂h/∂x  ≈  Sobel_x  (rate of height change left→right)
        ∂h/∂y  ≈  Sobel_y  (rate of height change top→bottom)

    The surface tangent vectors in X and Y directions are:
        T = (1, 0, ∂h/∂x · scale)
        B = (0, 1, ∂h/∂y · scale)

    The normal is their cross-product:
        N = T × B = (-∂h/∂x · scale,  -∂h/∂y · scale,  1)

    Then we normalise N to unit length:
        N̂ = N / |N|

    Finally, we map from [-1,+1] to [0,255] for PNG storage:
        pixel = (N̂ + 1) / 2 × 255

    Result interpretation:
        • Flat surface  → pixel (128, 128, 255)  (straight-up blue)
        • Left slope    → pixel (> 128, 128, ...)  (reddish)
        • Right slope   → pixel (< 128, 128, ...)  (dark red)

    Args:
        hmap         : float32 height map, shape (H, W), values [0,1]
        height_scale : amplifies gradient steepness (higher = sharper normals)

    Returns:
        normal_map   : float32 array shape (H, W, 3), values [0,1]
        normals_xyz  : float32 array shape (H, W, 3), raw unit vectors [-1,1]
    """
    # ── Sobel kernels (3×3) ───────────────────────────────────────────────
    # cv2.Sobel computes:  dst = ∑ K(i,j) · src(x+i, y+j)
    # ksize=3 uses the standard 3×3 Sobel operator
    # ddepth=cv2.CV_32F keeps floating-point precision
    dhdx = cv2.Sobel(hmap, cv2.CV_32F, 1, 0, ksize=3)   # ∂h/∂x
    dhdy = cv2.Sobel(hmap, cv2.CV_32F, 0, 1, ksize=3)   # ∂h/∂y

    # ── Scale gradients by the height amplification factor ────────────────
    dhdx *= height_scale
    dhdy *= height_scale

    # ── Build normal vectors N = (-∂h/∂x, -∂h/∂y, 1) ─────────────────────
    Nx = -dhdx
    Ny = -dhdy
    Nz = np.ones_like(hmap)   # always points "up"

    # ── Normalise to unit length ───────────────────────────────────────────
    length = np.sqrt(Nx**2 + Ny**2 + Nz**2) + 1e-8
    Nx /= length
    Ny /= length
    Nz /= length

    normals_xyz = np.stack([Nx, Ny, Nz], axis=-1)  # shape (H, W, 3) in [-1,1]

    # ── Encode to [0,1] for saving as PNG ─────────────────────────────────
    normal_map = (normals_xyz + 1.0) / 2.0           # remap [-1,1] → [0,1]

    print(f"[✓] Normal map computed: {hmap.shape[1]}×{hmap.shape[0]}  "
          f"channels=3 (R=X, G=Y, B=Z)")
    return normal_map.astype(np.float32), normals_xyz


# =============================================================================
# STEP 4 — BUILD 3D MESH FROM HEIGHT MAP
# =============================================================================

def heightmap_to_mesh(hmap: np.ndarray,
                      height_scale: float = 10.0) -> trimesh.Trimesh:
    """
    MESH CONSTRUCTION ALGORITHM:
    We treat the height map as a regular grid of (rows × cols) vertices.

    ┌──────────────────────────────────────────────────────────────────┐
    │  VERTEX LAYOUT (each cell = one grid point)                     │
    │                                                                  │
    │   (col, row) →  3D position:                                    │
    │       X = col / (cols-1) × width                                │
    │       Y = row / (rows-1) × depth                                │
    │       Z = hmap[row, col] × height_scale                         │
    │                                                                  │
    │  QUAD → 2 TRIANGLES:                                            │
    │                                                                  │
    │   v00 ──── v01        Triangle A:  v00, v01, v11                │
    │    │  ╲     │         Triangle B:  v00, v11, v10                │
    │    │   ╲    │                                                   │
    │   v10 ──── v11        (counter-clockwise = front-facing)        │
    └──────────────────────────────────────────────────────────────────┘

    UV COORDINATES:
        Each vertex also gets a UV coord for texture mapping:
            u = col / (cols-1)
            v = 1 - row / (rows-1)   ← flipped so image top = mesh top

    Args:
        hmap         : float32 height map (H, W)
        height_scale : vertical scale factor in 3D units

    Returns:
        mesh : trimesh.Trimesh object ready for export
    """
    rows, cols = hmap.shape
    print(f"[…] Building mesh: {cols}×{rows} = {cols*rows} vertices, "
          f"{2*(cols-1)*(rows-1)} triangles ...")

    # ── Build vertex positions ────────────────────────────────────────────
    # np.mgrid creates two 2D index arrays matching (row, col) positions
    row_idx, col_idx = np.mgrid[0:rows, 0:cols]

    x = col_idx.flatten().astype(np.float32) / (cols - 1)   # [0,1] across width
    y = row_idx.flatten().astype(np.float32) / (rows - 1)   # [0,1] across depth
    z = hmap.flatten() * 1                     # elevation

    vertices = np.stack([x, y, z], axis=-1)   # shape: (rows*cols, 3)

    # ── Build face indices (two triangles per quad cell) ──────────────────
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = r * cols + c    
            v01 = r * cols + (c + 1)   
            v10 = (r + 1) * cols + c    
            v11 = (r + 1) * cols + (c + 1)  

            faces.append([v00, v01, v11])

            faces.append([v00, v11, v10])

    faces = np.array(faces, dtype=np.int32)  

    u = col_idx.flatten().astype(np.float32) / (cols - 1)
    v = 1.0 - row_idx.flatten().astype(np.float32) / (rows - 1)
    uv = np.stack([u, v], axis=-1)         

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )

    # Attach UV as vertex attribute (metadata, used by some exporters)
    mesh.vertex_attributes["uv"] = uv

    print(f"[✓] Mesh built:  {len(vertices)} vertices,  {len(faces)} faces")
    return mesh

def save_normal_map(normal_map: np.ndarray, path: str) -> None:
    """
    Save the float32 normal map (values [0,1]) as an 8-bit RGB PNG.

    """
    arr_uint8 = np.clip(normal_map * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr_uint8, mode="RGB")
    pil_img.save(path)
    print(f"[✓] Normal map saved → {path}")


def save_mesh(mesh: trimesh.Trimesh, path: str) -> None:
    """
    Export the mesh as Wavefront .obj (plain text, universally supported).

    .obj files can be opened in:
        Blender, Maya, 3ds Max, MeshLab, Windows 3D Viewer, online viewers
    """
    mesh.export(path)
    print(f"[✓] Mesh saved    → {path}")

def visualize_results(hmap: np.ndarray,
                      normal_map: np.ndarray,
                      height_scale: float) -> None:
    """
    Show a 4-panel figure:
        [Top-left]     Original grayscale height map
        [Top-right]    Normal map (RGB encoded)
        [Bottom-left]  3D surface plot coloured by elevation
        [Bottom-right] 3D surface plot using normal map colours
    """
    rows, cols = hmap.shape

    # Create coordinate grids for the surface plot
    X = np.linspace(0, 1, cols)
    Y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(X, Y)
    Z = hmap * height_scale

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")  

    # ── Panel 1: Height Map ───────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(hmap, cmap="terrain", origin="upper")
    ax1.set_title("Height Map (Grayscale)", color="white", fontsize=11)
    ax1.axis("off")
    ax1.set_facecolor("#1a1a2e")

    # ── Panel 2: Normal Map ───────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(np.clip(normal_map, 0, 1), origin="upper")
    ax2.set_title("Normal Map (R=X, G=Y, B=Z)", color="white", fontsize=11)
    ax2.axis("off")
    ax2.set_facecolor("#1a1a2e")

    # ── Panel 3: 3D elevation surface ────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    surf = ax3.plot_surface(X, Y, Z, cmap="terrain",
                            linewidth=0, antialiased=True, alpha=0.9)
    ax3.set_title("3D Terrain (elevation colour)", color="white", fontsize=11)
    ax3.set_facecolor("#1a1a2e")
    ax3.tick_params(colors="white", labelsize=6)

    # ── Panel 4: 3D surface coloured by normal map ───────────────────────
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    # facecolors needs shape (rows-1, cols-1, 4) — use centre of each face
    face_colors = np.clip(normal_map[:-1, :-1], 0, 1)  
    # Pad alpha channel
    alpha_ch    = np.ones((*face_colors.shape[:2], 1), dtype=np.float32)
    face_colors = np.concatenate([face_colors, alpha_ch], axis=-1)

    ax4.plot_surface(X, Y, Z, facecolors=face_colors,
                    linewidth=0, antialiased=True)
    ax4.set_title("3D Terrain (normal colour)", color="white", fontsize=11)
    ax4.set_facecolor("#1a1a2e")
    ax4.tick_params(colors="white", labelsize=6)

    plt.suptitle("Image → Height Map → Normal Map → 3D Terrain",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print("[✓] Visualisation window opened.")

def main():
    print("=" * 60)
    print("  TERRAIN MESH GENERATOR")
    print("=" * 60)

    # ── Determine output directory (same folder as this script) ──────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path  = os.path.join(script_dir, INPUT_IMAGE)
    mesh_path   = os.path.join(script_dir, OUTPUT_MESH)
    normal_path = os.path.join(script_dir, OUTPUT_NORMAL)

    img_rgb = load_image(input_path)

    hmap = image_to_heightmap(img_rgb,
                               target_w=MESH_WIDTH,
                               target_h=MESH_HEIGHT,
                               sigma=SMOOTH_SIGMA)

    normal_map, normals_xyz = heightmap_to_normalmap(hmap, height_scale=2.0)

    mesh = heightmap_to_mesh(hmap, height_scale=HEIGHT_SCALE)

    save_normal_map(normal_map, normal_path)
    save_mesh(mesh, mesh_path)

    visualize_results(hmap, normal_map, HEIGHT_SCALE)

    print("\n" + "=" * 60)
    print("  DONE!")
    print(f"  Height map   : computed in memory")
    print(f"  Normal map   : {normal_path}")
    print(f"  3D mesh (.obj): {mesh_path}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()