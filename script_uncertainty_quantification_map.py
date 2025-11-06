#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# --------------------------------------------------------
# Physical + numerical parameters
# --------------------------------------------------------
nx, nz = 200, 200           # 200 × 200 spatial grid
x = np.linspace(0, 200, nx) # μm
z = np.linspace(0, 200, nz)
X, Z = np.meshgrid(x, z)

# --------------------------------------------------------
# True permittivity model (Healthy skin)
# Smooth hydration transition using erfc exactly like your paper
# --------------------------------------------------------
phi_surface = 0.20
phi_basal   = 0.70
d_SC = 15     # μm
w    = 5      # μm

def hydration(z):
    return phi_basal + (phi_surface - phi_basal) * erfc((z - d_SC)/w)

def eps_r(phi):
    # Dual-Debye, real part at 1 THz (your Table V values)
    eps_inf = 4.5
    Delta_eps1 = 78.4 * phi
    Delta_eps2 = 5.8  * phi
    tau1 = 8.3e-12
    tau2 = 0.3e-12
    nu   = 1e12
    omega = 2*np.pi*nu
    
    eps = eps_inf \
          + Delta_eps1/(1 + (omega*tau1)**2) \
          + Delta_eps2/(1 + (omega*tau2)**2)
    return eps

phi_map   = hydration(Z)
eps_true  = eps_r(phi_map)

# --------------------------------------------------------
# Uncertainty model (SNR = 40 dB)
# --------------------------------------------------------
SNR_dB = 40
epsilon_mean = np.mean(eps_true)

sigma_base = epsilon_mean / (10**(SNR_dB/20))     # noise level

# Spatial gradient magnitude (sensitivity ↑ with gradient)
grad_x = np.gradient(eps_true, axis=0)
grad_z = np.gradient(eps_true, axis=1)
grad_mag = np.sqrt(grad_x**2 + grad_z**2)

# Depth-dependent signal decay (penetration depth ~ 25 μm)
penetration_depth = 25
signal_strength = np.exp(-Z / penetration_depth)

# Combined uncertainty model
sigma_map = sigma_base * (1 + 0.5*grad_mag) / (signal_strength + 0.1)

# Boundary amplification (CPML & zero-sensitivity zones)
boundary_mask = (X < 10) | (X > 190) | (Z < 10) | (Z > 190)
sigma_map[boundary_mask] *= 5

# Clip to visually match your figure (0 → 0.005)
sigma_map = np.clip(sigma_map, 0, 0.005)

# --------------------------------------------------------
# FIGURE 3 — UNCERTAINTY MAP
# --------------------------------------------------------
plt.figure(figsize=(7,6))
levels = np.linspace(0, 0.005, 100)

im = plt.imshow(
    sigma_map.T,
    origin='lower',
    extent=[0, 200, 0, 200],
    cmap='viridis',
    vmin=0, vmax=0.005,
    aspect='auto'
)

# Interface marker
plt.axhline(15, color='white', linestyle='--', linewidth=1.5)
plt.text(2, 17, "SC interface (15 µm)",
         color='white', fontsize=10, weight='bold',
         bbox=dict(boxstyle='round', facecolor='black', alpha=0.4))

# Labels
plt.xlabel("x (µm)")
plt.ylabel("z (µm)")
plt.title("Uncertainty Map σ$_{\\epsilon_r}$(x, z) at SNR = 40 dB")

# Colorbar
cbar = plt.colorbar(im, label="σ$_{\\epsilon_r}$")

plt.tight_layout()
plt.show()
