#!/usr/bin/env python3
"""
Script 2 — Pathology Discrimination (Publication-ready)

Generates three panels:
(a) Healthy (baseline)
(b) Psoriasis (thickened SC, d_SC = 30 µm)
(c) Edema (φ_basal = 0.85)

Each panel shows:
- True ε_r(z) at 1 THz (solid)
- Mean reconstructed ε_r(z) from 50 noisy realizations (dashed)
- 95% confidence band (gray)

Also prints Mann–Whitney U-test p-values comparing basal-layer ε_r
(distribution across realizations) for:
- Healthy vs Psoriasis
- Healthy vs Edema

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu

# --------------------------
# Display / style
# --------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "mathtext.default": "regular"
})

# --------------------------
# Depth grid (µm)
# --------------------------
z = np.linspace(0.0, 200.0, 1000)  # µm

# --------------------------
# Hydration profile (error function)
# φ(z) = φ_basal + (φ_surface - φ_basal) * erfc((z - d_SC)/w)
# --------------------------
def hydration_profile(z_um, phi_surface, phi_basal, d_sc_um, w_um):
    return phi_basal + (phi_surface - phi_basal) * erfc((z_um - d_sc_um) / w_um)

# --------------------------
# ε_r(1 THz) calibrated mapping (consistent with manuscript)
# Targets: ε_r = 6.34 @ φ=0.20 and ε_r = 8.71 @ φ=0.70
# Linear calibration: ε_r(φ) = a + b * φ
# --------------------------
phi_low, eps_low = 0.20, 6.34
phi_high, eps_high = 0.70, 8.71
b_cal = (eps_high - eps_low) / (phi_high - phi_low)
a_cal = eps_low - b_cal * phi_low

def epsilon_r_1THz(phi):
    """Calibrated 1 THz permittivity consistent with your 6.34–8.71 band."""
    return a_cal + b_cal * phi

# --------------------------
# Pathology definitions
# --------------------------
cases = {
    "Healthy":   {"phi_s": 0.20, "phi_b": 0.70, "d_sc": 15.0, "w": 5.0},
    "Psoriasis": {"phi_s": 0.10, "phi_b": 0.60, "d_sc": 30.0, "w": 5.0},
    "Edema":     {"phi_s": 0.20, "phi_b": 0.85, "d_sc": 15.0, "w": 5.0},
}

# --------------------------
# Reconstruction emulator
# We emulate an adjoint+TV reconstruction by:
# 1) add Gaussian noise at chosen SNR (dB) to the "true" ε_r(z)
# 2) apply Savitzky–Golay smoothing as a TV-like denoiser
# --------------------------
SNR_dB = 40.0
N_realizations = 50
sg_window = 19   # odd
sg_poly   = 3

rng = np.random.default_rng(42)

def reconstruct_many(eps_true, nreal=50, snr_db=40.0):
    """
    Returns:
      rec_mean(z), rec_q025(z), rec_q975(z), basal_samples (N_realizations,)
    basal_samples are the average ε_r over a basal depth window (e.g., 60–100 µm).
    """
    # noise std from SNR: σ ≈ mean(|signal|) / 10^(SNR/20)
    sigma = np.mean(np.abs(eps_true)) / (10**(snr_db/20))

    recs = []
    basal_vals = []
    basal_mask = (z >= 60.0) & (z <= 100.0)

    for _ in range(nreal):
        noisy = eps_true + rng.normal(0.0, sigma, size=eps_true.size)
        # TV-like smoothing proxy
        smooth = savgol_filter(noisy, sg_window, sg_poly, mode='interp')
        recs.append(smooth)
        basal_vals.append(smooth[basal_mask].mean())

    recs = np.array(recs)  # (N_realizations, Nz)
    rec_mean = recs.mean(axis=0)
    rec_q025 = np.quantile(recs, 0.025, axis=0)
    rec_q975 = np.quantile(recs, 0.975, axis=0)
    basal_vals = np.array(basal_vals)

    return rec_mean, rec_q025, rec_q975, basal_vals

# --------------------------
# Compute profiles and reconstructions
# --------------------------
results = {}
for name, p in cases.items():
    phi = hydration_profile(z, p["phi_s"], p["phi_b"], p["d_sc"], p["w"])
    eps_true = epsilon_r_1THz(phi)
    rec_mean, rec_lo, rec_hi, basal_dist = reconstruct_many(eps_true, N_realizations, SNR_dB)
    results[name] = {
        "phi": phi,
        "eps_true": eps_true,
        "rec_mean": rec_mean,
        "rec_lo": rec_lo,
        "rec_hi": rec_hi,
        "basal_samples": basal_dist,
        "params": p
    }

# --------------------------
# Statistical tests (basal-layer distributions)
# --------------------------
H = results["Healthy"]["basal_samples"]
P = results["Psoriasis"]["basal_samples"]
E = results["Edema"]["basal_samples"]

u_hp, p_hp = mannwhitneyu(H, P, alternative='two-sided')
u_he, p_he = mannwhitneyu(H, E, alternative='two-sided')

print("="*70)
print("Mann–Whitney U-tests on basal-layer ε_r (averaged 60–100 µm)")
print(f"Healthy vs Psoriasis: U={u_hp:.1f}, p={p_hp:.3e}")
print(f"Healthy vs Edema:     U={u_he:.1f}, p={p_he:.3e}")
print("="*70)

# --------------------------
# Plotting
# --------------------------
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
titles = ["(a) Healthy (baseline)", "(b) Psoriasis (d$_{SC}$ = 30 µm)", "(c) Edema (φ$_{basal}$ = 0.85)"]

for ax, (title, name) in zip(axs, zip(titles, ["Healthy", "Psoriasis", "Edema"])):
    R = results[name]
    eps_true = R["eps_true"]
    rec_mean = R["rec_mean"]
    rec_lo   = R["rec_lo"]
    rec_hi   = R["rec_hi"]
    p        = R["params"]

    # 95% CI fill
    ax.fill_between(z, rec_lo, rec_hi, color='lightgray', alpha=0.8, label='95% CI (recon)')
    # Recon mean
    ax.plot(z, rec_mean, 'r--', lw=2.0, label='Reconstructed (mean)')
    # True
    ax.plot(z, eps_true, 'b-', lw=2.2, label=r'True $\epsilon_r(z)$')

    # SC thickness marker
    ax.axvline(p["d_sc"], color='darkgreen', ls='--', lw=1.5, alpha=0.8)
    ax.text(p["d_sc"]+2, eps_low+0.05, r'SC interface', color='darkgreen', fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(r'Depth $z$ ($\mu$m)')
    ax.grid(True, alpha=0.25)

axs[0].set_ylabel(r'$\epsilon_r$ @ 1 THz')
axs[0].legend(loc='lower right', framealpha=0.95)

# Tight layout & show
plt.suptitle("Pathology Discrimination: True vs Reconstructed Permittivity Profiles", y=1.04, fontsize=13)
plt.tight_layout()
plt.show()

