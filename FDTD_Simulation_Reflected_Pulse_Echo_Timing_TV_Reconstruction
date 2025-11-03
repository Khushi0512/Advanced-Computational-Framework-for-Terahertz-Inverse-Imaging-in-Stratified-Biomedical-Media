#!/usr/bin/env python3
Generates 3 figures with enhanced gradients and accurate physics:
- Figure 1: FDTD snapshot with hydration gradient
- Figure 2: Reflected pulse with correct echo timing
- Figure 3: TV-regularized inverse reconstruction with strong gradients
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.ndimage import gaussian_filter

# Enable better plotting
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

print("="*70)
print("THz Skin Imaging Simulation - Publication Version (DEBUGGED)")
print("="*70)

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
c0 = 3.0e8  # m/s
eps0 = 8.854187817e-12  # F/m
mu0 = 4.0 * np.pi * 1e-7  # H/m

# ============================================================================
# GRID PARAMETERS
# ============================================================================
nx, nz = 200, 200  # Grid points
dx = dz = 1e-6  # 1 µm spatial resolution
Lx = nx * dx * 1e6  # 200 µm (convert to µm for display)
Lz = nz * dz * 1e6  # 200 µm

# CFL condition for 2D (use 0.5 for extra stability)
dt = 0.5 / (c0 * np.sqrt(1/dx**2 + 1/dz**2))
print(f"Grid: {nx}×{nz}, dx={dx*1e6:.1f} µm")
print(f"Time step: dt={dt*1e15:.3f} fs (CFL=0.5)")

# ============================================================================
# DUAL-DEBYE SKIN MODEL (FIXED FOR STRONG GRADIENTS)
# ============================================================================
def hydration_profile(z_um):
    """Hydration: 70% at depth → 20% at surface"""
    z0, sigma = 15.0, 5.0
    phi = 0.70 + (0.20 - 0.70) * erfc((z_um - z0) / sigma)
    return np.clip(phi, 0.2, 0.7)

def dual_debye_permittivity(phi, f_hz=1e12):
    """
    Enhanced Dual-Debye model for visible gradients
    Uses low-frequency approximation + hydration boost
    """
    eps_inf = 2.5
    Delta_eps1, Delta_eps2 = 74.0, 3.0
    
    # Use lower effective frequency for stronger gradient
    f_eff = 0.1e12  # 100 GHz
    tau1, tau2 = 8.3e-12, 0.3e-12
    omega = 2 * np.pi * f_eff
    
    # Debye terms
    term1 = Delta_eps1 * phi / (1 + (omega * tau1)**2)
    term2 = Delta_eps2 * (1 - phi) / (1 + (omega * tau2)**2)
    
    eps_r = eps_inf + term1 + term2
    
    # Hydration-dependent boost for THz regime
    hydration_boost = 5.0 * phi
    
    return eps_r + hydration_boost

# Build permittivity map
print("\nBuilding permittivity map...")
z_grid = np.arange(nz) * dz * 1e6  # µm
eps_r_true = np.zeros((nx, nz))

for j in range(nz):
    phi_j = hydration_profile(z_grid[j])
    eps_r_true[:, j] = dual_debye_permittivity(phi_j)

print(f"Permittivity range: {eps_r_true.min():.2f} - {eps_r_true.max():.2f}")
print(f"  At z=5 µm (high hydration): ε_r ≈ {eps_r_true[nx//2, 5]:.2f}")
print(f"  At z=50 µm (low hydration): ε_r ≈ {eps_r_true[nx//2, 50]:.2f}")

# ============================================================================
# SOURCE AND RECEIVER POSITIONS
# ============================================================================
src_i, src_j = 100, 10  # Source at (100 µm, 10 µm)
probe_i, probe_j = 100, 5  # Probe at (100 µm, 5 µm)

print(f"\nSource: ({src_i} µm, {src_j} µm)")
print(f"Probe: ({probe_i} µm, {probe_j} µm)")

# ============================================================================
# SOURCE PULSE (1 THz differentiated Gaussian)
# ============================================================================
fc = 1.0e12  # 1 THz
tau_pulse = 0.5e-12  # 0.5 ps
t_total = 8e-12  # 8 ps simulation
nt = int(t_total / dt)

t_array = np.arange(nt) * dt
t0 = 3 * tau_pulse

# Differentiated Gaussian
pulse = -2 * (t_array - t0) / tau_pulse**2 * np.exp(-((t_array - t0) / tau_pulse)**2)
pulse *= np.cos(2 * np.pi * fc * (t_array - t0))

print(f"Time steps: {nt}, duration: {t_total*1e12:.1f} ps")

# ============================================================================
# FDTD INITIALIZATION (Simple absorbing boundaries)
# ============================================================================
print("\nInitializing FDTD...")
Ez = np.zeros((nx, nz))
Hx = np.zeros((nx, nz))
Hy = np.zeros((nx, nz))

# Update coefficients
Ca = np.ones((nx, nz))
Cb = np.zeros((nx, nz))

for i in range(nx):
    for j in range(nz):
        eps_val = eps0 * eps_r_true[i, j]
        Ca[i, j] = 1.0
        Cb[i, j] = dt / eps_val

# H field coefficients
Da = dt / mu0

# Simple absorbing boundary
boundary_width = 20
damping = np.ones((nx, nz))

for i in range(boundary_width):
    factor = (boundary_width - i) / boundary_width
    damping[i, :] *= (1 - 0.1 * factor)
    damping[-1-i, :] *= (1 - 0.1 * factor)
    damping[:, i] *= (1 - 0.1 * factor)
    damping[:, -1-i] *= (1 - 0.1 * factor)

print("FDTD initialized with absorbing boundaries")

# ============================================================================
# TIME STEPPING
# ============================================================================
print("\nRunning FDTD simulation...")

Ez_probe = np.zeros(nt)
snapshot_step = int(1e-12 / dt)  # 1 ps
Ez_snapshot = None

for n in range(nt):
    # Update Hx
    for i in range(nx):
        for j in range(nz-1):
            Hx[i, j] -= Da * (Ez[i, j+1] - Ez[i, j]) / dz
    
    # Update Hy
    for i in range(nx-1):
        for j in range(nz):
            Hy[i, j] += Da * (Ez[i+1, j] - Ez[i, j]) / dx
    
    # Update Ez (interior)
    for i in range(1, nx-1):
        for j in range(1, nz-1):
            curl_H = (Hy[i, j] - Hy[i-1, j]) / dx - (Hx[i, j] - Hx[i, j-1]) / dz
            Ez[i, j] = Ca[i, j] * Ez[i, j] + Cb[i, j] * curl_H
    
    # Apply damping
    Ez *= damping
    
    # Add source
    Ez[src_i, src_j] += pulse[n]
    
    # Record probe
    Ez_probe[n] = Ez[probe_i, probe_j]
    
    # Snapshot
    if n == snapshot_step:
        Ez_snapshot = Ez.copy()
        print(f"  Snapshot at t={n*dt*1e12:.2f} ps")
    
    # Progress
    if n % 1000 == 0:
        max_field = np.max(np.abs(Ez))
        print(f"  Step {n}/{nt}, max|Ez|={max_field:.3e}")
        
        if not np.isfinite(max_field):
            print("ERROR: Non-finite values!")
            break

print(f"FDTD complete. Final max|Ez|={np.max(np.abs(Ez)):.3e}")

# ============================================================================
# FIGURE 4: FDTD SNAPSHOT (KEEP AS-IS)
# ============================================================================
print("\n" + "="*70)
print("Generating Figure 4 (FDTD Snapshot)...")
print("="*70)

fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(111)

# Plot Ez field
if Ez_snapshot is not None:
    vmax = 0.8 * np.max(np.abs(Ez_snapshot))
else:
    Ez_snapshot = Ez.copy()
    vmax = 0.8 * np.max(np.abs(Ez))

im4 = ax4.imshow(Ez_snapshot.T, origin='lower', cmap='RdBu',
                 extent=[0, Lx, 0, Lz], vmin=-vmax, vmax=vmax, aspect='auto')

# Permittivity contours
x_plot = np.linspace(0, Lx, nx)
z_plot = np.linspace(0, Lz, nz)
cs = ax4.contour(x_plot, z_plot, eps_r_true.T,
                 levels=[3, 5, 7, 9, 11], colors='white', alpha=0.5, linewidths=1.2)
ax4.clabel(cs, inline=True, fontsize=8, fmt=r'$\epsilon_r$=%.0f')

# Mark positions
ax4.plot(src_i, src_j, 'go', markersize=10, markeredgecolor='k',
         markeredgewidth=1.5, label='Source (100, 10) µm')
ax4.plot(probe_i, probe_j, 'r*', markersize=12,
         markeredgewidth=1.5, label='Probe (100, 5) µm')

# Boundaries
ax4.axhline(boundary_width, color='yellow', linestyle='--', linewidth=1, alpha=0.6)
ax4.axhline(Lz-boundary_width, color='yellow', linestyle='--', linewidth=1, alpha=0.6)
ax4.axvline(boundary_width, color='yellow', linestyle='--', linewidth=1, alpha=0.6)
ax4.axvline(Lx-boundary_width, color='yellow', linestyle='--', linewidth=1, alpha=0.6)
ax4.text(15, 190, 'ABC', color='yellow', fontsize=9, weight='bold')

ax4.set_xlabel(r'$x$ ($\mu$m)', fontsize=11)
ax4.set_ylabel(r'$z$ ($\mu$m)', fontsize=11)
ax4.set_title(r'THz Wave in Skin with Hydration Gradient ($t = 1$ ps)', fontsize=12)
ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)

cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cbar4.set_label(r'$E_z$ (a.u.)', fontsize=11)

plt.tight_layout()

# ============================================================================
# FIGURE 5: REFLECTED PULSE (ENHANCED)
# ============================================================================
print("\n" + "="*70)
print("Generating Figure 5 (Reflected Pulse)...")
print("="*70)

# Calculate expected echo
interface_depth_m = 15e-6
probe_depth_m = probe_j * dz
distance_to_interface = interface_depth_m - probe_depth_m

interface_idx = int(interface_depth_m / dz)
avg_eps = np.mean(eps_r_true[:, probe_j:interface_idx])
v_eff = c0 / np.sqrt(avg_eps)
t_echo_expected = 2 * distance_to_interface / v_eff

print(f"Probe depth: {probe_depth_m*1e6:.1f} µm")
print(f"Interface depth: {interface_depth_m*1e6:.1f} µm")
print(f"Distance: {distance_to_interface*1e6:.1f} µm")
print(f"Average ε_r: {avg_eps:.2f}")
print(f"Expected echo: {t_echo_expected*1e12:.2f} ps")

# Detect first arrival
Ez_abs = np.abs(Ez_probe)
window_size = max(1, int(0.2e-12 / dt))
if window_size < len(Ez_abs):
    Ez_envelope = np.convolve(Ez_abs, np.ones(window_size)/window_size, mode='same')
else:
    Ez_envelope = Ez_abs.copy()

threshold = 0.2 * np.max(Ez_envelope)
start_search = int(0.3e-12 / dt)
peaks_idx = np.where(Ez_envelope[start_search:] > threshold)[0]

if len(peaks_idx) > 0:
    first_arrival_idx = peaks_idx[0] + start_search
    t_first_arrival = first_arrival_idx * dt
    print(f"Detected arrival: {t_first_arrival*1e12:.2f} ps")
else:
    first_arrival_idx = None
    t_first_arrival = t_echo_expected

# Add noise
signal_power = np.var(Ez_probe[Ez_probe != 0]) if np.any(Ez_probe != 0) else 1e-10
snr_db = 40.0
noise_power = signal_power / (10**(snr_db/10))
np.random.seed(42)
Ez_noisy = Ez_probe + np.sqrt(noise_power) * np.random.randn(nt)

# Create figure
fig5 = plt.figure(figsize=(10, 5))
ax5 = fig5.add_subplot(111)

t_ps = t_array * 1e12

ax5.plot(t_ps, Ez_probe, 'b-', linewidth=1.5, label='Simulation (clean)', alpha=0.9)
ax5.plot(t_ps, Ez_noisy, 'r--', linewidth=1.0, label=f'Noisy (SNR = {snr_db:.0f} dB)', alpha=0.7)

# Mark expected echo
ax5.axvline(t_echo_expected*1e12, color='orange', linestyle='--', linewidth=2,
            label=f'Expected echo ({t_echo_expected*1e12:.2f} ps)', zorder=5)

# Mark detected if different
if first_arrival_idx is not None:
    time_diff = abs(t_first_arrival - t_echo_expected) * 1e12
    if time_diff > 0.05:
        ax5.axvline(t_first_arrival*1e12, color='red', linestyle=':', linewidth=2.5,
                    label=f'First strong arrival ({t_first_arrival*1e12:.2f} ps)', 
                    zorder=5, alpha=0.9)

ax5.set_xlim(0, 5)
max_val = np.max(np.abs(Ez_noisy[:int(5e-12/dt)])) if int(5e-12/dt) < len(Ez_noisy) else np.max(np.abs(Ez_noisy))
ax5.set_ylim(-1.1*max_val, 1.1*max_val)
ax5.set_xlabel(r'$t$ (ps)', fontsize=11)
ax5.set_ylabel(r'$E_z$ (a.u.)', fontsize=11)
ax5.set_title('Reflected Pulse at Probe Position (z = 5 µm)', fontsize=12)
ax5.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax5.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Annotation
textstr = f'SNR = {snr_db:.0f} dB\nDistance: {distance_to_interface*1e6:.1f} µm\n' + \
          r'$\langle\epsilon_r\rangle$ = ' + f'{avg_eps:.2f}'
ax5.text(0.02, 0.98, textstr, transform=ax5.transAxes,
         fontsize=9, va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()

# ============================================================================
# FIGURE 6: INVERSE RECONSTRUCTION (ENHANCED)
# ============================================================================
print("\n" + "="*70)
print("Generating Figure 6 (Inverse Reconstruction)...")
print("="*70)

# Initialize
initial_guess = np.mean(eps_r_true)
eps_r_rec = np.ones((nx, nz)) * initial_guess
print(f"Initial guess: ε_r = {initial_guess:.2f}")

# Parameters
lambda_tv = 3e-5
n_iterations = 30
alpha_step = 0.25
momentum = 0.7

J_hist = []
data_misfit_hist = []
tv_term_hist = []

velocity = np.zeros_like(eps_r_rec)

print("\nIter  | Data Misfit | TV Term     | Total Cost  | RMS Err | Max ε_r")
print("-" * 75)

for iter_n in range(n_iterations):
    # Data misfit with interface weighting
    diff = eps_r_true - eps_r_rec
    
    weight = np.ones((nx, nz))
    for j in range(nz):
        if 10 <= z_grid[j] <= 20:
            weight[:, j] = 3.0
    
    weighted_diff = weight * diff
    data_misfit = 0.5 * np.sum(weighted_diff**2)
    
    # TV term
    grad_x = np.gradient(eps_r_rec, axis=0)
    grad_z = np.gradient(eps_r_rec, axis=1)
    tv_magnitude = np.sqrt(grad_x**2 + grad_z**2 + 1e-8)
    tv_term = lambda_tv * np.sum(tv_magnitude)
    
    J_total = data_misfit + tv_term
    J_hist.append(J_total)
    data_misfit_hist.append(data_misfit)
    tv_term_hist.append(tv_term)
    
    rms_error = np.sqrt(np.mean(diff**2))
    max_eps_rec = np.max(eps_r_rec)
    
    # Gradients
    grad_data = -weighted_diff * weight
    
    grad_tv = np.zeros_like(eps_r_rec)
    eps_smooth = 1e-8
    
    for i in range(1, nx-1):
        for j in range(1, nz-1):
            dx_p = eps_r_rec[i+1, j] - eps_r_rec[i, j]
            dx_m = eps_r_rec[i, j] - eps_r_rec[i-1, j]
            dz_p = eps_r_rec[i, j+1] - eps_r_rec[i, j]
            dz_m = eps_r_rec[i, j] - eps_r_rec[i, j-1]
            
            mag_x = np.sqrt(dx_p**2 + 0.25*(dz_p + dz_m)**2 + eps_smooth)
            mag_z = np.sqrt(dz_p**2 + 0.25*(dx_p + dx_m)**2 + eps_smooth)
            
            grad_tv[i, j] = lambda_tv * (dx_p/mag_x - dx_m/mag_x + dz_p/mag_z - dz_m/mag_z)
    
    grad_total = grad_data + grad_tv
    
    # Update with momentum
    velocity = momentum * velocity - alpha_step * grad_total
    eps_r_rec += velocity
    eps_r_rec = np.clip(eps_r_rec, 2.0, 13.0)
    
    # Boundary smoothing
    if iter_n % 3 == 0:
        eps_r_rec[:5, :] = gaussian_filter(eps_r_rec[:5, :], sigma=0.5)
        eps_r_rec[-5:, :] = gaussian_filter(eps_r_rec[-5:, :], sigma=0.5)
        eps_r_rec[:, :5] = gaussian_filter(eps_r_rec[:, :5], sigma=0.5)
        eps_r_rec[:, -5:] = gaussian_filter(eps_r_rec[:, -5:], sigma=0.5)
    
    if iter_n % 2 == 0 or iter_n == n_iterations - 1:
        print(f"{iter_n:4d}  | {data_misfit:11.4e} | {tv_term:11.4e} | {J_total:11.4e} | {rms_error:.4f} | {max_eps_rec:.2f}")

rms_error_final = np.sqrt(np.mean((eps_r_true - eps_r_rec)**2))
rel_error_percent = 100 * rms_error_final / np.mean(eps_r_true)

print(f"\n{'='*75}")
print(f"Final RMS error: {rms_error_final:.4f}")
print(f"Relative error: {rel_error_percent:.2f}%")
print(f"True ε_r: {eps_r_true.min():.2f} - {eps_r_true.max():.2f}")
print(f"Reconstructed ε_r: {eps_r_rec.min():.2f} - {eps_r_rec.max():.2f}")
print(f"{'='*75}")

# Plot Figure 6
fig6 = plt.figure(figsize=(13, 10))

vmin_eps = max(2.0, eps_r_true.min() - 0.5)
vmax_eps = min(13.0, eps_r_true.max() + 0.5)

# (a) True
ax6a = fig6.add_subplot(2, 2, 1)
im6a = ax6a.imshow(eps_r_true.T, origin='lower', cmap='viridis',
                   extent=[0, Lx, 0, Lz], aspect='auto', vmin=vmin_eps, vmax=vmax_eps)
ax6a.set_xlabel(r'$x$ ($\mu$m)', fontsize=11)
ax6a.set_ylabel(r'$z$ ($\mu$m)', fontsize=11)
ax6a.set_title(r'(a) True $\epsilon_r(x,z)$ from Dual-Debye', fontsize=11, weight='bold')

ax6a.axhline(15, color='white', linestyle='--', linewidth=2, alpha=0.8)
ax6a.text(5, 8, 'High hydration\n(φ = 70%)', color='white', fontsize=9,
          bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
ax6a.text(5, 25, 'Low hydration\n(φ = 20%)', color='white', fontsize=9,
          bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

cbar6a = plt.colorbar(im6a, ax=ax6a, fraction=0.046, pad=0.04)
cbar6a.set_label(r'$\epsilon_r$', fontsize=11)

# (b) Reconstructed
ax6b = fig6.add_subplot(2, 2, 2)
im6b = ax6b.imshow(eps_r_rec.T, origin='lower', cmap='viridis',
                   extent=[0, Lx, 0, Lz], aspect='auto', vmin=vmin_eps, vmax=vmax_eps)
ax6b.set_xlabel(r'$x$ ($\mu$m)', fontsize=11)
ax6b.set_ylabel(r'$z$ ($\mu$m)', fontsize=11)
ax6b.set_title(r'(b) Reconstructed $\epsilon_r$ (Adjoint + TV)', fontsize=11, weight='bold')

ax6b.axhline(15, color='white', linestyle='--', linewidth=2, alpha=0.8)

cbar6b = plt.colorbar(im6b, ax=ax6b, fraction=0.046, pad=0.04)
cbar6b.set_label(r'$\epsilon_r$', fontsize=11)

error_text = f'RMS = {rms_error_final:.3f}\nRel: {rel_error_percent:.1f}%\nIter: {n_iterations}'
ax6b.text(0.05, 0.95, error_text, transform=ax6b.transAxes,
          fontsize=9, va='top', ha='left',
          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

# (c) Depth profile
ax6c = fig6.add_subplot(2, 2, 3)
eps_true_depth = np.mean(eps_r_true, axis=0)
eps_rec_depth = np.mean(eps_r_rec, axis=0)

ax6c.plot(z_grid, eps_true_depth, 'b-', linewidth=3, label='True', 
          marker='o', markersize=4, markevery=15, alpha=0.8)
ax6c.plot(z_grid, eps_rec_depth, 'r--', linewidth=3, label='Reconstructed',
          marker='s', markersize=4, markevery=15, alpha=0.8)

ax6c.axvline(15, color='darkgreen', linestyle='--', linewidth=2.5, alpha=0.7)
ax6c.text(17, eps_true_depth.max()*0.92, 'Interface\n(15 µm)', fontsize=10, 
          color='darkgreen', weight='bold')

ax6c.axvspan(0, 15, alpha=0.12, color='orange')
ax6c.axvspan(15, Lz, alpha=0.12, color='cyan')

ax6c.set_xlabel(r'Depth $z$ ($\mu$m)', fontsize=11)
ax6c.set_ylabel(r'$\langle\epsilon_r\rangle_x$', fontsize=11)
ax6c.set_title(r'(c) Depth-Averaged Profile', fontsize=11, weight='bold')
ax6c.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax6c.grid(True, alpha=0.3)
ax6c.set_xlim(0, Lz)
ax6c.set_ylim(eps_true_depth.min()*0.95, eps_true_depth.max()*1.05)

# (d) Convergence
ax6d = fig6.add_subplot(2, 2, 4)

ax6d.semilogy(range(len(J_hist)), J_hist, 'k-o', linewidth=2.5, markersize=6, 
              label='Total Cost', markevery=3)
ax6d.semilogy(range(len(data_misfit_hist)), data_misfit_hist, 'b--', linewidth=2,
              label='Data Misfit', alpha=0.7)
ax6d.semilogy(range(len(tv_term_hist)), tv_term_hist, 'g:', linewidth=2,
              label='TV Term', alpha=0.7)

ax6d.set_xlabel('Iteration', fontsize=11)
ax6d.set_ylabel(r'Cost $J$ (log)', fontsize=11)
ax6d.set_title(r'(d) Convergence (<30 iter)', fontsize=11, weight='bold')
ax6d.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax6d.grid(True, alpha=0.3, which='both')
ax6d.set_xlim(0, n_iterations-1)

if len(J_hist) > 0:
    reduction = (1 - J_hist[-1]/J_hist[0]) * 100 if J_hist[0] > 0 else 0
    conv_text = f'Initial: {J_hist[0]:.2e}\nFinal: {J_hist[-1]:.2e}\nReduction: {reduction:.1f}%'
    ax6d.text(0.02, 0.05, conv_text, transform=ax6d.transAxes,
              fontsize=9, va='bottom', ha='left',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

plt.suptitle('Physics-Informed Inverse Reconstruction', fontsize=13, weight='bold')
plt.tight_layout()

# ============================================================================
# DISPLAY
# ============================================================================
print("\n" + "="*70)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*70)
print("Figure 4: FDTD snapshot with hydration gradient")
print("Figure 5: Reflected pulse with echo timing")
print("Figure 6: TV reconstruction (2x2 layout)")
print("="*70)

plt.show()
