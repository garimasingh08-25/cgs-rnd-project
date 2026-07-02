#!/usr/bin/env python3
"""
terrain_cnn.py  —  CNN-Based 2D Image → 3D Terrain Pipeline
═══════════════════════════════════════════════════════════════════════════════

WHY CNN INSTEAD OF CLASSICAL METHODS?
───────────────────────────────────────
Classical (terrain_generator.py):
  • Gaussian blur       → fixed bell-curve filter (same for every image)
  • Sobel gradients     → 3×3 fixed edge kernel

CNN approach (this file):
  • HeightNet (U-Net)   → learns what pixel features predict height
  • NormalNet (ResNet)  → learns multi-scale surface orientation
  • Self-supervised     → no labelled training data needed; losses come
                          from physics-based constraints on the image itself

RESULT:
  ✓  Sharper ridges and valleys (edge-aware training)
  ✓  Smoother flat regions (TV regularisation)
  ✓  Physically consistent shading (Lambertian loss)
  ✓  Better normal map accuracy (integrability constraint)

HOW TO RUN IN VS CODE
──────────────────────
  1.  Ctrl + `  to open the integrated terminal
  2.  Activate venv:   terrain_env\\Scripts\\activate  (Windows)
                       source terrain_env/bin/activate  (Mac/Linux)
  3.  Install deps:    pip install -r requirements_cnn.txt
  4.  Run:
        python terrain_cnn.py photo.jpg
        python terrain_cnn.py landscape.png --epochs 500 --preview
        python terrain_cnn.py rock.jpg --scale-z 4 --strength 8
        python terrain_cnn.py big.jpg --train-size 128 --epochs 200  (faster)
═══════════════════════════════════════════════════════════════════════════════
"""

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  IMPORTS                                                                    │
# │                                                                             │
# │  Standard library (no install needed):                                      │
# │    os, sys, argparse, time                                                  │
# │                                                                             │
# │  Third-party (pip install -r requirements_cnn.txt):                         │
# │    torch        — neural networks, tensors, autograd                        │
# │    numpy        — array math                                                │
# │    Pillow       — image I/O                                                 │
# │    scipy        — classical Gaussian (used only for comparison output)      │
# │    tqdm         — progress bars                                             │
# │    matplotlib   — preview PNG  (optional, --preview flag)                   │
# └─────────────────────────────────────────────────────────────────────────────┘

import os
import sys
import argparse
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw):          # graceful fallback if tqdm not installed
        return x


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1  —  NEURAL NETWORK BUILDING BLOCKS                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class ConvBNReLU(nn.Module):
    """
    The fundamental CNN unit: Conv2d → BatchNorm2d → ReLU

    ┌─────────────────────────────────────────────────────────────┐
    │  CONVOLUTION  (Conv2d)                                      │
    │                                                             │
    │  A learnable k×k filter slides across the feature map.     │
    │  At each position (i, j):                                   │
    │                                                             │
    │    y[i,j] = Σ_{m=0}^{k-1} Σ_{n=0}^{k-1} W[m,n]·x[i+m,j+n] + b  │
    │                                                             │
    │  W = weight matrix  (learned by gradient descent)           │
    │  b = bias scalar    (learned by gradient descent)           │
    │                                                             │
    │  WHY BETTER THAN FIXED FILTERS?                             │
    │  Fixed filters (Sobel, Gaussian) do ONE thing.              │
    │  A Conv layer with C_out filters learns C_out different     │
    │  patterns simultaneously, each tuned to this image.         │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  BATCH NORMALISATION  (BatchNorm2d)                         │
    │                                                             │
    │  For each channel c, normalise across the spatial H×W:      │
    │                                                             │
    │    μ_c   = mean of activations in channel c                 │
    │    σ²_c  = variance of activations in channel c             │
    │                                                             │
    │    y_c = γ_c · (x_c - μ_c) / √(σ²_c + ε)  +  β_c         │
    │                                                             │
    │  γ, β are learnable scale/shift parameters.                  │
    │                                                             │
    │  BENEFITS:                                                  │
    │  • Prevents internal covariate shift                        │
    │  • Allows 10× higher learning rates                         │
    │  • Acts as regulariser → reduces need for Dropout           │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  RELU  (Rectified Linear Unit)                              │
    │                                                             │
    │    ReLU(x) = max(0, x)                                      │
    │                                                             │
    │  DERIVATIVE:                                                │
    │    dReLU/dx = 1  if x > 0                                   │
    │               0  if x ≤ 0                                   │
    │                                                             │
    │  WHY NOT SIGMOID/TANH?                                      │
    │  Sigmoid saturates at 0 and 1 → gradient ≈ 0 → vanishing   │
    │  ReLU has full gradient for x>0 → faster, deeper training   │
    └─────────────────────────────────────────────────────────────┘
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual Block: two ConvBNReLU layers + a SKIP CONNECTION

    ┌─────────────────────────────────────────────────────────────┐
    │  RESIDUAL LEARNING  (He et al., Deep Residual Networks)     │
    │                                                             │
    │  Standard block:   H(x) = F(x)                             │
    │  Residual block:   H(x) = F(x) + x                         │
    │                                                             │
    │  The network learns the RESIDUAL  F(x) = H(x) − x          │
    │  instead of the full mapping H(x).                          │
    │                                                             │
    │  GRADIENT FLOW:                                             │
    │  ∂Loss/∂x = ∂Loss/∂H · (∂F/∂x + I)                        │
    │                           ↑ identity                        │
    │  The identity term means gradients ALWAYS have a path       │
    │  back to early layers — no vanishing gradient.              │
    │                                                             │
    │  INTUITION FOR TERRAIN:                                     │
    │  If a region of the height map is already correct, F(x) → 0 │
    │  and H(x) ≈ x (pass-through).  The network only learns      │
    │  corrections, not the full signal from scratch.             │
    └─────────────────────────────────────────────────────────────┘

    Architecture:
        x ──► ConvBNReLU ──► ConvBNReLU ──► (+) ──► output
        │                                    ↑
        └────────────── skip ────────────────┘
    """
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x    # residual addition (skip connection)


class DoubleConv(nn.Module):
    """Two ConvBNReLU layers in sequence — standard U-Net encoder/decoder unit."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2  —  HEIGHTNET  (U-Net architecture)                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class HeightNet(nn.Module):
    """
    U-Net for self-supervised height map estimation.

    ┌─────────────────────────────────────────────────────────────┐
    │  U-NET ARCHITECTURE  (Ronneberger et al., 2015)             │
    │                                                             │
    │  Input  (1 × H × W)   ← grayscale image                    │
    │    │                                                        │
    │    ├──► E1: DoubleConv(1→32)   32 × H    × W    ──────────┐│
    │    ↓ MaxPool(2×2)                                           ││
    │    ├──► E2: DoubleConv(32→64)  64 × H/2  × W/2  ────────┐  ││
    │    ↓ MaxPool(2×2)                                         │  ││
    │    ├──► E3: DoubleConv(64→128) 128× H/4  × W/4  ──────┐  │  ││
    │    ↓ MaxPool(2×2)                                       │  │  ││
    │    └──► Bottleneck: 256 × H/8 × W/8                    │  │  ││
    │         ↑ Up + concat(E3)                               │  │  ││
    │         D3: DoubleConv(384→128)  128 × H/4 × W/4  ─────┘  │  ││
    │         ↑ Up + concat(E2)                                   │  ││
    │         D2: DoubleConv(192→64)    64 × H/2 × W/2  ─────────┘  ││
    │         ↑ Up + concat(E1)                                       ││
    │         D1: DoubleConv(96→32)     32 × H   × W   ──────────────┘│
    │         ↑ Conv(32→1) + Sigmoid                                   │
    │  Output (1 × H × W)  ← height map in [0, 1]  ───────────────────┘
    │                                                             │
    │  RECEPTIVE FIELD:                                           │
    │  With 3 pooling levels, a bottleneck neuron "sees" the      │
    │  equivalent of an ~64×64 pixel region in the input.         │
    │  This is why U-Net captures global structure AND detail.    │
    │                                                             │
    │  WHY SKIP CONNECTIONS?                                      │
    │  MaxPool loses spatial precision. Skip connections           │
    │  re-inject the high-resolution encoder features into        │
    │  the decoder, recovering fine detail after upsampling.      │
    └─────────────────────────────────────────────────────────────┘

    UPSAMPLING METHOD: Bilinear interpolation (not transposed convolution)
        For each output pixel at (i, j), bilinear interpolation computes:
            f(i,j) = (1-a)(1-b)·f(i0,j0) + a(1-b)·f(i1,j0)
                   + (1-a)b  ·f(i0,j1) + ab  ·f(i1,j1)
        where (a, b) is the fractional offset into the source pixel.
        No learnable parameters → stable, artefact-free upsampling.
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        # Encoder
        self.enc1 = DoubleConv(1,    c)     # 32
        self.enc2 = DoubleConv(c,    c*2)   # 64
        self.enc3 = DoubleConv(c*2,  c*4)   # 128
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck = DoubleConv(c*4, c*8)  # 256
        # Decoder (after concat, in_ch = up_ch + skip_ch)
        self.dec3 = DoubleConv(c*8 + c*4, c*4)  # 256+128 → 128
        self.dec2 = DoubleConv(c*4 + c*2, c*2)  # 128+64  → 64
        self.dec1 = DoubleConv(c*2 + c,   c)    # 64+32   → 32
        # Final head: 1×1 conv + Sigmoid → [0,1]
        self.head = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder path ──────────────────────────────────────────────────
        e1 = self.enc1(x)             # 32 × H    × W
        e2 = self.enc2(self.pool(e1)) # 64 × H/2  × W/2
        e3 = self.enc3(self.pool(e2)) # 128× H/4  × W/4
        b  = self.bottleneck(self.pool(e3))  # 256 × H/8 × W/8

        # ── Decoder path (upsample → concat skip → DoubleConv) ───────────
        d3 = self.dec3(self._upsample_concat(b,  e3))  # 128
        d2 = self.dec2(self._upsample_concat(d3, e2))  # 64
        d1 = self.dec1(self._upsample_concat(d2, e1))  # 32

        return self.head(d1)  # 1 × H × W  in [0, 1]

    @staticmethod
    def _upsample_concat(x: torch.Tensor,
                         skip: torch.Tensor) -> torch.Tensor:
        """
        Bilinear upsample x to match skip's spatial size,
        then concatenate along the channel dimension.

        concat(upsample(x), skip) shape:
            channels: x.C + skip.C
            H, W    : skip.H, skip.W
        """
        x = F.interpolate(x, size=skip.shape[2:],
                          mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3  —  NORMALNET  (Residual CNN)                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class NormalNet(nn.Module):
    """
    Residual CNN that refines Sobel-derived normals.

    ┌─────────────────────────────────────────────────────────────┐
    │  WHY CNN FOR NORMALS? (vs Sobel)                            │
    │                                                             │
    │  Sobel uses a fixed 3×3 neighbourhood:                      │
    │    ∂H/∂x ≈ (h[i,j+1] − h[i,j-1]) / 2                       │
    │                                                             │
    │  Problems:                                                  │
    │    • 3×3 receptive field → only sees immediate neighbours   │
    │    • No noise suppression in flat regions                   │
    │    • Staircase artefacts at diagonal edges                  │
    │                                                             │
    │  NormalNet uses 3 residual blocks, each with 3×3 convs:     │
    │    Effective receptive field after 6 layers = 13×13 pixels  │
    │    → smoother normals with multi-pixel context              │
    │                                                             │
    │  RESIDUAL DESIGN:                                           │
    │    CNN output is ADDED to the Sobel normals (not replaced). │
    │    The CNN only learns what Sobel gets wrong → faster, more │
    │    stable training.                                         │
    └─────────────────────────────────────────────────────────────┘

    Architecture:
        height_map (1×H×W) ──► ConvBNReLU ──► ResBlock ×3 ──► Conv(3) ──► refinement
        analytical_normals (3×H×W) ─────────────────────────────────────► (+) ──► L2 norm ──► output
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.init_conv = ConvBNReLU(1, base_ch)
        self.res1 = ResidualBlock(base_ch)
        self.res2 = ResidualBlock(base_ch)
        self.res3 = ResidualBlock(base_ch)
        # Output 3 channels: residual for (Nx, Ny, Nz)
        self.head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

    def forward(self, height: torch.Tensor,
                analytical: torch.Tensor) -> torch.Tensor:
        """
        height     : (1, 1, H, W)   ← CNN height map
        analytical : (1, 3, H, W)   ← Sobel normals in [-1, 1]
        returns    : (1, 3, H, W)   ← refined unit normals
        """
        feat     = self.init_conv(height)
        feat     = self.res1(feat)
        feat     = self.res2(feat)
        feat     = self.res3(feat)
        residual = self.head(feat)              # learned correction

        # Small step: don't overwhelm the good Sobel baseline
        normals  = analytical + 0.15 * residual

        # L2-normalise: make each vector a unit vector (length = 1)
        # For vector N = (Nx, Ny, Nz):
        #   N_hat = N / ||N||_2   where ||N||_2 = sqrt(Nx² + Ny² + Nz²)
        return F.normalize(normals, dim=1, p=2)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4  —  LOSS FUNCTIONS  (self-supervised training)                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def sobel_gradients(x: torch.Tensor):
    """
    Compute spatial gradients using Sobel kernels.

    ┌─────────────────────────────────────────────────────────────┐
    │  SOBEL OPERATOR                                             │
    │                                                             │
    │  Approximates the first-order derivative (slope) of a       │
    │  discrete function using a 3×3 weighted kernel.             │
    │                                                             │
    │  X kernel (∂f/∂x):      Y kernel (∂f/∂y):                  │
    │   ┌──────────────┐       ┌──────────────┐                   │
    │   │ -1   0   +1 │       │ -1  -2  -1  │                   │
    │   │ -2   0   +2 │       │  0   0   0  │                   │
    │   │ -1   0   +1 │       │ +1  +2  +1  │                   │
    │   └──────────────┘       └──────────────┘                   │
    │                                                             │
    │  The centre row/column is weighted by 2 for smoothing.      │
    │  This makes it a separable operator:                         │
    │    Kx = [1, 2, 1]ᵀ · [-1, 0, 1]    (outer product)        │
    │                                                             │
    │  Applied as 2D convolution:                                 │
    │    gx[i,j] = Σ_{m,n} Kx[m,n] · f[i+m, j+n]                │
    └─────────────────────────────────────────────────────────────┘
    """
    # Build kernels on the same device/dtype as x
    kx = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Total Variation Loss — promotes piecewise-smooth outputs.

    ┌─────────────────────────────────────────────────────────────┐
    │  TOTAL VARIATION  (Rudin, Osher, Fatemi, 1992)              │
    │                                                             │
    │  Original (isotropic TV):                                   │
    │    TV(H) = (1/N) Σ_{i,j} √[(H[i+1,j]−H[i,j])²            │
    │                             + (H[i,j+1]−H[i,j])²]          │
    │                                                             │
    │  We use the differentiable anisotropic version (L2):        │
    │    L_TV = mean(|∂H/∂x|²) + mean(|∂H/∂y|²)                  │
    │                                                             │
    │  INTUITION:                                                 │
    │    • Large ∂H/∂x → abrupt horizontal change → penalised     │
    │    • Flat terrain: both derivatives ≈ 0 → L_TV = 0         │
    │    • A spiky, noisy terrain → L_TV >> 0 → network avoids it │
    │                                                             │
    │  EFFECT ON HEIGHT MAP:                                      │
    │    Noisy pixel spikes are smoothed out.                     │
    │    Genuine slopes (large smooth gradients) are preserved.   │
    └─────────────────────────────────────────────────────────────┘
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]   # ∂H/∂x  (horizontal diff)
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]   # ∂H/∂y  (vertical diff)
    return dx.pow(2).mean() + dy.pow(2).mean()


def gradient_consistency_loss(height: torch.Tensor,
                               image: torch.Tensor) -> torch.Tensor:
    """
    Gradient Consistency Loss — align height gradients with image edges.

    ┌─────────────────────────────────────────────────────────────┐
    │  INTUITION                                                  │
    │                                                             │
    │  Bright-to-dark boundaries in the image usually correspond  │
    │  to terrain features: ridges, cliffs, valleys.              │
    │  This loss encourages height changes where the image has    │
    │  edges, and discourages height changes in uniform regions.  │
    │                                                             │
    │  EDGE-WEIGHTED FORMULA:                                     │
    │    gx_H, gy_H = Sobel(H)         height gradients           │
    │    gx_I, gy_I = Sobel(I)         image  gradients           │
    │    w = |∇I| = √(gx_I² + gy_I²)  edge magnitude             │
    │                                                             │
    │    L_grad = mean(w · |gx_H − gx_I|²)                       │
    │           + mean(w · |gy_H − gy_I|²)                        │
    │                                                             │
    │  The weight w amplifies the loss where the image has strong │
    │  edges — ensuring the height map respects image boundaries. │
    └─────────────────────────────────────────────────────────────┘
    """
    gx_h, gy_h = sobel_gradients(height)
    gx_i, gy_i = sobel_gradients(image)

    # Edge magnitude as weight (clamp to avoid log(0) issues)
    edge_w = (gx_i.pow(2) + gy_i.pow(2)).sqrt().clamp(0.01, 1.0)

    loss_x = (edge_w * (gx_h - gx_i).pow(2)).mean()
    loss_y = (edge_w * (gy_h - gy_i).pow(2)).mean()
    return loss_x + loss_y


def laplacian_smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Laplacian (second-order) smoothness loss.

    ┌─────────────────────────────────────────────────────────────┐
    │  DISCRETE LAPLACIAN OPERATOR                                │
    │                                                             │
    │  Approximates ∂²f/∂x² + ∂²f/∂y² (sum of second derivatives)│
    │  using a 3×3 kernel:                                        │
    │                                                             │
    │    ┌──────────────┐                                         │
    │    │  0   1   0  │                                         │
    │    │  1  -4   1  │    Δf[i,j] ≈ f[i-1,j]+f[i+1,j]         │
    │    │  0   1   0  │              +f[i,j-1]+f[i,j+1]−4f[i,j] │
    │    └──────────────┘                                         │
    │                                                             │
    │  L_lap = mean(|Δh|²)                                        │
    │                                                             │
    │  PHYSICAL MEANING:                                          │
    │    • Δh = 0 everywhere → harmonic function → perfectly      │
    │      smooth interpolation between boundary values           │
    │    • Δh ≠ 0 → sharp bends, creases → penalised             │
    │                                                             │
    │  TV vs LAPLACIAN:                                           │
    │    TV (1st order)  → discourages steep slopes               │
    │    Laplacian (2nd) → discourages abrupt slope changes       │
    │    Together → smooth slopes that curve gently               │
    └─────────────────────────────────────────────────────────────┘
    """
    k = torch.tensor([[0.,  1., 0.],
                       [1., -4., 1.],
                       [0.,  1., 0.]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    lap = F.conv2d(x, k, padding=1)
    return lap.pow(2).mean()


def shading_consistency_loss(height: torch.Tensor,
                              image: torch.Tensor,
                              strength: float = 5.0) -> torch.Tensor:
    """
    Lambertian shading consistency loss.

    ┌─────────────────────────────────────────────────────────────┐
    │  LAMBERTIAN REFLECTANCE MODEL                               │
    │                                                             │
    │  For a matte (non-shiny) surface, the observed brightness   │
    │  depends only on the angle between the surface normal N      │
    │  and the light direction L:                                  │
    │                                                             │
    │    I_rendered = max(0, N · L)   ← dot product               │
    │                                                             │
    │  ALGORITHM:                                                 │
    │    1. Compute surface normals N from height H (Sobel):      │
    │         gx = ∂H/∂x   gy = ∂H/∂y                            │
    │         N_raw = (−gx·s, −gy·s, 1)                           │
    │         N     = N_raw / ||N_raw||                           │
    │                                                             │
    │    2. Define light direction (upper-left source):           │
    │         L = normalise(0.5, 0.5, −1.0)                       │
    │                                                             │
    │    3. Rendered image:                                       │
    │         I_r = ReLU(N · L) = max(0, Nx·Lx + Ny·Ly + Nz·Lz) │
    │                                                             │
    │    4. Loss = MSE(I_r, I_input)                              │
    │         L_shade = mean((I_r − I)²)                          │
    │                                                             │
    │  WHY THIS HELPS:                                            │
    │    If bright pixels should correspond to surfaces facing     │
    │    the light (shallower angle → more light), then the height │
    │    map must have normals consistent with that observation.   │
    └─────────────────────────────────────────────────────────────┘
    """
    # Step 1: compute normals from height
    gx, gy = sobel_gradients(height)
    nx  = -gx * strength
    ny  = -gy * strength
    nz  = torch.ones_like(nx)
    mag = (nx.pow(2) + ny.pow(2) + nz.pow(2)).sqrt().clamp(min=1e-6)
    nx, ny, nz = nx / mag, ny / mag, nz / mag

    # Step 2: normalise light direction L = (0.5, 0.5, -1) / |L|
    lx = torch.tensor( 0.5, dtype=height.dtype, device=height.device)
    ly = torch.tensor( 0.5, dtype=height.dtype, device=height.device)
    lz = torch.tensor(-1.0, dtype=height.dtype, device=height.device)
    l_mag = (lx**2 + ly**2 + lz**2).sqrt()
    lx, ly, lz = lx / l_mag, ly / l_mag, lz / l_mag

    # Step 3: Lambertian shading (clamp negative dot products to 0)
    rendered = (nx * lx + ny * ly + nz * lz).clamp(0.0, 1.0)

    # Step 4: MSE against input image
    return F.mse_loss(rendered, image)


def normal_consistency_loss(normals: torch.Tensor,
                             height: torch.Tensor,
                             strength: float = 5.0) -> torch.Tensor:
    """
    Normal-height integrability constraint.

    ┌─────────────────────────────────────────────────────────────┐
    │  INTEGRABILITY CONDITION                                     │
    │                                                             │
    │  For a surface with normal N = (Nx, Ny, Nz):                │
    │                                                             │
    │    The slope in X: ∂H/∂x = −Nx / Nz                        │
    │    The slope in Y: ∂H/∂y = −Ny / Nz                        │
    │                                                             │
    │  Rearranged (multiply both sides by Nz):                    │
    │    Nz · ∂H/∂x + Nx = 0                                      │
    │    Nz · ∂H/∂y + Ny = 0                                      │
    │                                                             │
    │  LOSS:                                                      │
    │    L_int = mean((Nz·gx·s + Nx)²) + mean((Nz·gy·s + Ny)²)  │
    │                                                             │
    │  This forces the predicted normals to be geometrically      │
    │  consistent with the height map gradients.                  │
    └─────────────────────────────────────────────────────────────┘
    """
    gx, gy = sobel_gradients(height)
    nx = normals[:, 0:1]
    ny = normals[:, 1:2]
    nz = normals[:, 2:3]

    consist_x = (nz * gx * strength + nx).pow(2).mean()
    consist_y = (nz * gy * strength + ny).pow(2).mean()
    unit_loss  = (normals.pow(2).sum(dim=1, keepdim=True) - 1.0).pow(2).mean()

    return consist_x + consist_y + 5.0 * unit_loss


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5  —  ANALYTICAL NORMALS  (Sobel baseline for NormalNet)         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def compute_analytical_normals(h_tensor: torch.Tensor,
                                strength: float = 5.0) -> torch.Tensor:
    """
    Compute tangent-space normals from height map using Sobel gradients.

    FORMULA:
        gx, gy = Sobel(H)
        N_raw  = (−gx·s,  −gy·s,  1)
        N      = N_raw / ||N_raw||_2

    Returns tensor of shape (1, 3, H, W) in range [−1, 1].
    """
    gx, gy = sobel_gradients(h_tensor)
    nx = -gx * strength
    ny = -gy * strength
    nz = torch.ones_like(nx)
    raw = torch.cat([nx, ny, nz], dim=1)            # (1, 3, H, W)
    return F.normalize(raw, dim=1, p=2)             # unit vectors


def normals_to_rgb(n_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert float normal tensor (1, 3, H, W) in [−1, 1]
    to uint8 RGB (H, W, 3) in [0, 255].

    Encoding:  R = (Nx+1)/2 · 255
               G = (Ny+1)/2 · 255
               B = (Nz+1)/2 · 255
    """
    n = n_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return ((n + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6  —  TRAINING LOOPS                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def _resize_for_training(image_np: np.ndarray,
                         max_side: int) -> np.ndarray:
    """
    Resize image so the longest side ≤ max_side, preserving aspect ratio.
    CNNs are fully convolutional — trained weights work on any size at inference.
    """
    h, w = image_np.shape
    if max(h, w) <= max_side:
        return image_np
    scale = max_side / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    pil = Image.fromarray((image_np * 255).astype(np.uint8))
    small = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(small, dtype=np.float32) / 255.0


def train_height_net(image_np: np.ndarray, args, device) -> HeightNet:
    """
    Self-supervised training of HeightNet on a single image.

    ┌─────────────────────────────────────────────────────────────┐
    │  SELF-SUPERVISED LEARNING                                   │
    │                                                             │
    │  No ground-truth height maps exist.                         │
    │  Instead, we optimise physics-based constraints:             │
    │                                                             │
    │    L_total = λ_tv   · L_TV                                  │
    │            + λ_grad · L_gradient_consistency                │
    │            + λ_lap  · L_laplacian                           │
    │            + λ_shade· L_shading                             │
    │                                                             │
    │  DEEP IMAGE PRIOR EFFECT:                                   │
    │  A randomly initialised U-Net acts as a natural-image       │
    │  regulariser. Optimising it on a single image recovers      │
    │  structure rather than noise (Ulyanov et al., 2018).        │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  ADAM OPTIMISER                                             │
    │                                                             │
    │  Adaptive Moment Estimation (Kingma & Ba, 2015):            │
    │                                                             │
    │    m_t = β₁·m_{t−1} + (1−β₁)·∇L      ← 1st moment (mean)  │
    │    v_t = β₂·v_{t−1} + (1−β₂)·∇L²     ← 2nd moment (var)   │
    │    m̂   = m_t / (1 − β₁ᵗ)             ← bias correction     │
    │    v̂   = v_t / (1 − β₂ᵗ)                                   │
    │    θ_t = θ_{t−1} − α · m̂ / (√v̂ + ε)  ← weight update      │
    │                                                             │
    │  Defaults: α=1e-3, β₁=0.9, β₂=0.999, ε=1e-8               │
    │                                                             │
    │  WHY ADAM?                                                  │
    │  Each parameter gets its own adaptive learning rate,         │
    │  scaled by the magnitude of recent gradients.               │
    │  Parameters that rarely update (low gradient) get           │
    │  relatively larger steps. This speeds convergence.           │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  COSINE ANNEALING LR SCHEDULE                               │
    │                                                             │
    │  lr(t) = lr_min + (lr_max − lr_min) · (1 + cos(πt/T)) / 2  │
    │                                                             │
    │  Starts at lr_max, smoothly decays to lr_min over T epochs. │
    │  Benefits: fast progress early, fine-tuning later.          │
    └─────────────────────────────────────────────────────────────┘
    """
    img_small = _resize_for_training(image_np, args.train_size)
    h_s, w_s  = img_small.shape
    img_t     = torch.tensor(img_small, dtype=torch.float32, device=device)
    img_t     = img_t.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

    model     = HeightNet(base_ch=args.base_ch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    print(f"      HeightNet  |  {w_s}×{h_s}  |  {args.epochs} epochs  |  {device}")

    best_loss  = float("inf")
    best_state = None

    loop = tqdm(range(args.epochs), desc="      HeightNet", unit="ep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  loss={postfix[0]:.4f}  [{elapsed}]",
                postfix=[0.0]) if HAS_TQDM else range(args.epochs)

    for epoch in (loop if HAS_TQDM else range(args.epochs)):
        optimizer.zero_grad()
        pred = model(img_t)          # forward pass → (1, 1, H, W)

        # Compute individual losses
        l_tv    = total_variation_loss(pred)
        l_grad  = gradient_consistency_loss(pred, img_t)
        l_lap   = laplacian_smoothness_loss(pred)
        l_shade = shading_consistency_loss(pred, img_t, args.strength)

        # Weighted combination
        loss = (args.w_tv    * l_tv
              + args.w_grad  * l_grad
              + args.w_lap   * l_lap
              + args.w_shade * l_shade)

        loss.backward()   # backpropagation: compute ∂Loss/∂θ for all θ

        # Gradient clipping: if ||∂L/∂θ|| > max_norm, scale it down
        # Prevents "exploding gradients" which cause NaN weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()    # update θ ← θ − lr · ∂L/∂θ  (Adam-adjusted)
        scheduler.step()    # decay learning rate

        l_val = loss.item()
        if l_val < best_loss:
            best_loss  = l_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if HAS_TQDM:
            loop.postfix[0] = l_val

        if not HAS_TQDM and (epoch + 1) % 50 == 0:
            print(f"      epoch {epoch+1:4d}/{args.epochs}  loss={l_val:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    return model


def infer_height(model: HeightNet, image_np: np.ndarray,
                 device) -> np.ndarray:
    """
    Run trained HeightNet on the full-resolution image.

    FULLY CONVOLUTIONAL INFERENCE:
        Training used a small image for speed.
        At inference the same weights are applied to the full image.
        Because Conv2d operates independently at each position,
        the spatial size at inference can differ from training.
        Only BatchNorm statistics are size-independent (running mean/var).
    """
    model.eval()
    with torch.no_grad():
        t  = torch.tensor(image_np, dtype=torch.float32, device=device)
        t  = t.unsqueeze(0).unsqueeze(0)
        h  = model(t).squeeze().cpu().numpy()
    lo, hi = h.min(), h.max()
    return (h - lo) / (hi - lo) if hi > lo else h


def train_normal_net(height_np: np.ndarray, args, device) -> NormalNet:
    """
    Self-supervised training of NormalNet.

    Loss = normal_consistency (integrability) + TV smoothness
    """
    h_t = torch.tensor(height_np, dtype=torch.float32, device=device)
    h_t = h_t.unsqueeze(0).unsqueeze(0)                  # (1,1,H,W)
    analytical = compute_analytical_normals(h_t, args.strength)

    model     = NormalNet(base_ch=args.base_ch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.normal_epochs, eta_min=args.lr * 0.005
    )

    print(f"      NormalNet  |  {args.normal_epochs} epochs  |  {device}")

    best_loss  = float("inf")
    best_state = None

    loop = tqdm(range(args.normal_epochs), desc="      NormalNet", unit="ep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  loss={postfix[0]:.4f}  [{elapsed}]",
                postfix=[0.0]) if HAS_TQDM else range(args.normal_epochs)

    for epoch in (loop if HAS_TQDM else range(args.normal_epochs)):
        optimizer.zero_grad()
        normals  = model(h_t, analytical)

        l_cons   = normal_consistency_loss(normals, h_t, args.strength)
        l_smooth = (total_variation_loss(normals[:, 0:1])
                  + total_variation_loss(normals[:, 1:2])
                  + total_variation_loss(normals[:, 2:3]))
        loss     = l_cons + 0.01 * l_smooth

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        l_val = loss.item()
        if l_val < best_loss:
            best_loss  = l_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if HAS_TQDM:
            loop.postfix[0] = l_val

    model.load_state_dict(best_state)
    model.eval()
    return model


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7  —  OBJ MESH GENERATION  (same math, better input)            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def height_to_obj(height: np.ndarray, obj_path: str,
                  scale_xy: float = 10.0, scale_z: float = 2.0,
                  step: int = 2, mtl_stem: str = None):
    """
    Triangulated Wavefront OBJ mesh from height field.

    VERTEX POSITION FORMULA:
        X = (col/(cols−1) − 0.5) × scale_xy
        Y = height[row,col] × scale_z
        Z = (row/(rows−1) − 0.5) × scale_xy

    VERTEX NORMAL FORMULA (central differences):
        ∂Y/∂X = (h[r,c+1]−h[r,c−1]) / (2·dx)
        ∂Y/∂Z = (h[r+1,c]−h[r−1,c]) / (2·dz)
        N = normalise(−∂Y/∂X, 1, −∂Y/∂Z)

    TRIANGULATION:  2 CCW triangles per quad
        Triangle 1: TL → BL → TR
        Triangle 2: TR → BL → BR
    """
    h_map      = height[::step, ::step]
    rows, cols = h_map.shape
    n_v        = rows * cols
    n_t        = (rows - 1) * (cols - 1) * 2
    print(f"      Grid: {cols}×{rows}   Vertices: {n_v:,}   Triangles: {n_t:,}")

    xs  = (np.arange(cols, dtype=np.float64) / (cols - 1) - 0.5) * scale_xy
    zs  = (np.arange(rows, dtype=np.float64) / (rows - 1) - 0.5) * scale_xy
    ys  = h_map.astype(np.float64) * scale_z
    us  = np.arange(cols, dtype=np.float64) / (cols - 1)
    vs  = 1.0 - np.arange(rows, dtype=np.float64) / (rows - 1)

    dx  = scale_xy / (cols - 1)
    dz  = scale_xy / (rows - 1)
    pad = np.pad(ys, 1, mode="edge")
    dydx = (pad[1:-1, 2:] - pad[1:-1, :-2]) / (2.0 * dx)
    dydz = (pad[2:, 1:-1] - pad[:-2, 1:-1]) / (2.0 * dz)
    vnx  = -dydx
    vny  = np.ones((rows, cols), dtype=np.float64)
    vnz  = -dydz
    vmag = np.maximum(np.sqrt(vnx**2 + vny**2 + vnz**2), 1e-8)
    vnx /= vmag;  vny /= vmag;  vnz /= vmag

    with open(obj_path, "w", buffering=1 << 16) as f:
        f.write("# CNN terrain_cnn.py — 3D terrain mesh\n")
        if mtl_stem:
            f.write(f"mtllib {mtl_stem}.mtl\n")
        f.write("g terrain\n")
        if mtl_stem:
            f.write("usemtl terrain_mat\n")
        f.write("\n")
        lines = []
        for r in range(rows):
            for c in range(cols):
                lines.append(f"v {xs[c]:.6f} {ys[r,c]:.6f} {zs[r]:.6f}\n")
        f.writelines(lines);  f.write("\n")
        lines = []
        for r in range(rows):
            for c in range(cols):
                lines.append(f"vt {us[c]:.6f} {vs[r]:.6f}\n")
        f.writelines(lines);  f.write("\n")
        lines = []
        for r in range(rows):
            for c in range(cols):
                lines.append(f"vn {vnx[r,c]:.6f} {vny[r,c]:.6f} {vnz[r,c]:.6f}\n")
        f.writelines(lines);  f.write("\n")
        lines = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                tl = r*cols+c+1;    tr = r*cols+c+2
                bl = (r+1)*cols+c+1; br = (r+1)*cols+c+2
                lines.append(f"f {tl}/{tl}/{tl} {bl}/{bl}/{bl} {tr}/{tr}/{tr}\n")
                lines.append(f"f {tr}/{tr}/{tr} {bl}/{bl}/{bl} {br}/{br}/{br}\n")
        f.writelines(lines)

    if mtl_stem:
        mtl_path = os.path.join(os.path.dirname(obj_path), f"{mtl_stem}.mtl")
        with open(mtl_path, "w") as mf:
            mf.write("newmtl terrain_mat\n")
            mf.write("Ka 0.2 0.2 0.2\nKd 0.6 0.6 0.6\nKs 0.1 0.1 0.1\nNs 16\n")
            mf.write(f"map_Kd {mtl_stem}_height_cnn.png\n")
            mf.write(f"map_bump {mtl_stem}_normal_cnn.png\n")
    return n_v, n_t


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8  —  COMPARISON PREVIEW                                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def render_preview(raw, height_classical, height_cnn, normal_cnn, out_path):
    """
    5-panel comparison:
        Row 1:  [① Source]  [② Classical Height]  [③ CNN Height]
        Row 2:  [④ CNN Normal]  [⑤ 3-D CNN Surface]
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LightSource

        BG = "#0d1117"
        fig = plt.figure(figsize=(22, 10), facecolor=BG)
        fig.suptitle("CNN Terrain Pipeline  —  Classical vs CNN Comparison",
                     color="#e6edf3", fontsize=15, fontweight="bold", y=0.97)
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.3, wspace=0.18)

        ax1  = fig.add_subplot(gs[0, 0])
        ax2  = fig.add_subplot(gs[0, 1])
        ax3  = fig.add_subplot(gs[0, 2])
        ax4  = fig.add_subplot(gs[1, 0])
        ax3d = fig.add_subplot(gs[1, 1:], projection="3d")

        def _p(ax, title, sub=""):
            ax.set_title(f"{title}\n{sub}", color="#8b949e", fontsize=10, pad=5)
            ax.axis("off");  ax.set_facecolor(BG)

        ax1.imshow(raw, cmap="gray");  _p(ax1, "① Source Image")

        im2 = ax2.imshow(height_classical, cmap="terrain", vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, fraction=0.05, pad=0.03)
        _p(ax2, "② Classical Height", "Gaussian Blur")

        im3 = ax3.imshow(height_cnn, cmap="terrain", vmin=0, vmax=1)
        plt.colorbar(im3, ax=ax3, fraction=0.05, pad=0.03)
        _p(ax3, "③ CNN Height", "U-Net self-supervised")

        ax4.imshow(normal_cnn);  _p(ax4, "④ CNN Normal Map", "NormalNet + integrability")

        sub  = max(1, min(height_cnn.shape) // 100)
        hs   = height_cnn[::sub, ::sub]
        R, C = hs.shape
        X, Z = np.meshgrid(np.linspace(-1, 1, C), np.linspace(-1, 1, R))
        ls   = LightSource(azdeg=315, altdeg=45)
        surf = ls.shade(hs, cmap=plt.cm.terrain, vert_exag=3.0, blend_mode="soft")
        ax3d.plot_surface(X, Z, hs, facecolors=surf,
                          linewidth=0, antialiased=True, shade=False)
        ax3d.set_facecolor(BG);  ax3d.set_axis_off()
        ax3d.view_init(elev=40, azim=-50)
        ax3d.set_title("⑤ 3-D CNN Terrain", color="#8b949e", fontsize=10, pad=5)

        fig.patch.set_facecolor(BG)
        plt.savefig(out_path, dpi=130, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        plt.close(fig)
        print(f"      Saved → {out_path}")

    except ImportError:
        print("      [skip] pip install matplotlib  to enable preview")
    except Exception as e:
        print(f"      [skip] Preview error: {e}")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9  —  MAIN PIPELINE                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def _kb(path: str) -> None:
    print(f"      Saved  → {path}   ({os.path.getsize(path)/1024:.1f} KB)")


def process(args) -> None:
    os.makedirs(args.output, exist_ok=True)
    stem   = os.path.splitext(os.path.basename(args.input))[0]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    BAR = "═" * 62
    print(f"\n{BAR}")
    print(f"  terrain_cnn.py  ·  {os.path.basename(args.input)}")
    print(f"  device={device}  ·  epochs={args.epochs}")
    print(f"{BAR}\n")
    t0 = time.perf_counter()

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("[1/5]  Loading …")
    raw_gray = np.array(Image.open(args.input).convert("L"),
                        dtype=np.float32) / 255.0
    H, W = raw_gray.shape
    print(f"       {W}×{H} px\n")

    # ── 2. Classical baseline (for comparison only) ───────────────────────
    print("[2/5]  Classical height map (comparison baseline) …")
    h_cl = gaussian_filter(raw_gray, sigma=2.0)
    lo, hi = h_cl.min(), h_cl.max()
    h_cl = (h_cl - lo) / (hi - lo) if hi > lo else h_cl

    # ── 3. CNN Height Map ─────────────────────────────────────────────────
    print(f"\n[3/5]  CNN height map (U-Net, self-supervised) …")
    h_model  = train_height_net(raw_gray, args, device)
    h_cnn    = infer_height(h_model, raw_gray, device)
    hm_path  = os.path.join(args.output, f"{stem}_height_cnn.png")
    Image.fromarray((h_cnn * 255).clip(0, 255).astype(np.uint8), "L").save(hm_path)
    _kb(hm_path)

    # ── 4. CNN Normal Map ─────────────────────────────────────────────────
    print(f"\n[4/5]  CNN normal map (NormalNet, residual refinement) …")
    n_model = train_normal_net(h_cnn, args, device)
    h_t     = torch.tensor(h_cnn, dtype=torch.float32,
                            device=device).unsqueeze(0).unsqueeze(0)
    anal    = compute_analytical_normals(h_t, args.strength)
    with torch.no_grad():
        n_t = n_model(h_t, anal)
    normal_rgb = normals_to_rgb(n_t)
    nm_path    = os.path.join(args.output, f"{stem}_normal_cnn.png")
    Image.fromarray(normal_rgb, "RGB").save(nm_path)
    _kb(nm_path)

    # ── 5. OBJ Mesh ───────────────────────────────────────────────────────
    print(f"\n[5/5]  3D terrain mesh …")
    obj_path = os.path.join(args.output, f"{stem}_terrain_cnn.obj")
    height_to_obj(h_cnn, obj_path,
                  scale_xy=args.scale_xy, scale_z=args.scale_z,
                  step=args.step,
                  mtl_stem=(stem + "_cnn") if args.mtl else None)
    _kb(obj_path)

    # ── Preview ────────────────────────────────────────────────────────────
    if args.preview:
        print("\n[+]  Comparison preview …")
        prev = os.path.join(args.output, f"{stem}_cnn_preview.png")
        render_preview(raw_gray, h_cl, h_cnn, normal_rgb, prev)

    elapsed = time.perf_counter() - t0
    print(f"\n{BAR}")
    print(f"  ✓  Done in {elapsed:.1f} s")
    print(f"  Outputs: {os.path.abspath(args.output)}")
    print(f"{BAR}\n")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10  —  CLI                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    p = argparse.ArgumentParser(
        prog="terrain_cnn.py",
        description="CNN-based 2D image → height map + normal map + 3D OBJ terrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  python terrain_cnn.py photo.jpg
  python terrain_cnn.py landscape.png --epochs 500 --preview
  python terrain_cnn.py rock.jpg --scale-z 4 --strength 8
  python terrain_cnn.py big.jpg --train-size 128 --epochs 150  (fast)

OUTPUTS  (for photo.jpg):
  output/photo_height_cnn.png     ← CNN height map
  output/photo_normal_cnn.png     ← CNN normal map
  output/photo_terrain_cnn.obj    ← 3D mesh (import into Blender / Unity)
  output/photo_cnn_preview.png    ← comparison (--preview)
        """)

    p.add_argument("input", help="Input image (JPG PNG BMP TIFF WebP)")
    p.add_argument("-o", "--output", default="./output")

    g = p.add_argument_group("CNN training")
    g.add_argument("--epochs",        type=int,   default=300,
                   help="HeightNet training epochs [300]")
    g.add_argument("--normal-epochs", type=int,   default=150,
                   help="NormalNet training epochs [150]")
    g.add_argument("--lr",            type=float, default=1e-3,
                   help="Learning rate [0.001]")
    g.add_argument("--base-ch",       type=int,   default=32,
                   help="U-Net base channels [32]")
    g.add_argument("--train-size",    type=int,   default=256,
                   help="Training image size (longest side) [256]")
    g.add_argument("--cpu",           action="store_true",
                   help="Force CPU (default: use CUDA if available)")

    g = p.add_argument_group("loss weights")
    g.add_argument("--w-tv",    type=float, default=0.01,
                   help="Total variation loss weight [0.01]")
    g.add_argument("--w-grad",  type=float, default=1.0,
                   help="Gradient consistency weight [1.0]")
    g.add_argument("--w-lap",   type=float, default=0.005,
                   help="Laplacian smoothness weight [0.005]")
    g.add_argument("--w-shade", type=float, default=0.5,
                   help="Shading consistency weight [0.5]")

    g = p.add_argument_group("terrain output")
    g.add_argument("--strength", type=float, default=5.0,
                   help="Normal map strength [5.0]")
    g.add_argument("--scale-xy", type=float, default=10.0)
    g.add_argument("--scale-z",  type=float, default=2.0)
    g.add_argument("--step",     type=int,   default=2,
                   help="Mesh stride: 1=full, 2=half, 4=quarter [2]")
    g.add_argument("--mtl",      action="store_true",
                   help="Write .mtl material file")
    g.add_argument("--preview",  action="store_true",
                   help="Save 5-panel comparison PNG (needs matplotlib)")

    args = p.parse_args()
    if not os.path.isfile(args.input):
        print(f"Error: not found — {args.input}", file=sys.stderr)
        sys.exit(1)
    process(args)


if __name__ == "__main__":
    main()