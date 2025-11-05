#!/usr/bin/env python3
"""
Script 2 — Pathology Discrimination (publication-ready)

Generates a 3-panel plot of permittivity-vs-depth at 1 THz for:
(a) Healthy:      d_SC = 15 µm, φ_surface=0.20, φ_basal=0.70
(b) Psoriasis:    d_SC = 30 µm, φ_surface=0.10, φ_basal=0.60
(c) Edema:        d_SC = 15 µm, φ_surface=0.20, φ_basal=0.85

Features:
- Hydration profile phi(z) via error-function transition
- Calibrated ε_r(φ) mapping to match paper’s 1 THz band (6.34–8.71)
- 50 noisy “inverse reconstructions” with TV-like post-smoothing
- 95% confidence intervals (shaded)
- Mann–Whitney U tests (printed) on the deep/basal region

Requires: numpy, matplotlib, scipy (special.erfc, ndimage.gaussian_filter1d, stats.mannwhitneyu)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

# -----------------------------
# Plot look
# -----------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})

# -----------------------------
# Depth grid (µm)
# -----------------------------
z_um = np.linspace(0, 200, 801)  # 0..200 µm, 0.25 µm step

# -----------------------------
# Hydration profile (error-function)
# phi(z) = φ_b + (φ_s - φ_b) * erfc((z - d_SC)/w)
# -----------------------------
def hydration_profile(z_um, phi_surface, phi_basal, d_SC_um, w_um=5.0):
    phi = phi_basal + (phi_surface - phi_basal) * erfc((z_um - d_SC_um)/w_um)
    # Clamp to [0,1] physically
    return np.clip(phi, 0.0, 1.0)

# -----------------------------
# Calibrated ε_r(φ) at 1 THz
# Rationale:
#   Your manuscript specifies ε_r ∈ [6.34, 8.71] for φ ∈ [0.20, 0.70] at 1 THz.
#   To remain consistent across all figures, we use a linear calibration that
#   reproduces those endpoints exactly (physics-informed calibration for 1 THz band).
# -----------------------------
def epsilon_r_1THz_calibrated(phi):
    # Map phi in [0.20, 0.70] → ε_r in [6.34, 8.71] (linear), extrapolate softly outside
    phi0, phi1 = 0.20, 0.70
    eps0, eps1 = 6.34, 8.71
    a = (eps1 - eps0) / (phi1 - phi0)        # slope
    b = eps0 - a * phi0                      # intercept
    eps = a * phi + b
    # Soft clamp to reasonable clinical band
    return np.clip(eps, 3.0, 10.0)

# -----------------------------
# Pathology definitions
# -----------------------------
cases = [
    dict(name="Healthy",   phi_s=0.20, phi_b=0.70, d_sc=15.0, color="#1f77b4"),
    dict(name="Psoriasis", phi_s=0.10, phi_b=0.60, d_sc=30.0, color="#ff7f0e"),
    dict(name="Edema",     phi_s=0.20, phi_b=0.85, d_sc=15.0, color="#2ca02c"),
]

# -----------------------------
# "Inverse reconstruction" emulator:
#   Add white noise to ε_r(z), then light TV-like smoothing (1D Gaussian)
#   SNR_dB controls noise level (relative to dynamic range).
#   Returns N realizations array of shape (Nreal, Nz)
# -----------------------------
def emulate_inverse_recon(eps_true, z_um, Nreal=50, SNR_dB=40.0, tv_sigma_um=1.0, random_seed=7):
    rng = np.random.default_rng(random_seed)
    Nz = eps_true.size
    # Noise std as fraction of dynamic range (convert dB → linear amplitude ratio)
    dyn = np.max(eps_true) - np.min(eps_true) + 1e-12
    amp_ratio = 10**(-SNR_dB/20)   # e.g., 40 dB -> 0.01
    sigma_noise = amp_ratio * dyn

    # Samples at 0.25 µm spacing → sigma in samples:
    dz = z_um[1] - z_um[0]
    sigma_samples = max(0.0, tv_sigma_um / dz)  # Gaussian std in samples

    recons = []
    for _ in range(Nreal):
        noisy = eps_true + rng.normal(0.0, sigma_noise, size=Nz)
        # "TV-like" smoothing proxy (gentle)
        smooth = gaussian_filter1d(noisy, sigma=sigma_samples, mode="nearest")
        recons.append(smooth)
    return np.vstack(recons)  # (Nreal, Nz)

# -----------------------------
# Build true profiles and reconstructions
# -----------------------------
profiles_true = []
profiles_recon = []

Nreal = 50
SNR_dB = 40.0
tv_sigma_um = 1.0

for case in cases:
    phi = hydration_profile(z_um, case["phi_s"], case["phi_b"], case["d_sc"], w_um=5.0)
    eps_true = epsilon_r_1THz_calibrated(phi)
    profiles_true.append(eps_true)

    # Emulate 50 inverse reconstructions with noise + smoothing
    recons = emulate_inverse_recon(
        eps_true, z_um, Nreal=Nreal, SNR_dB=SNR_dB, tv_sigma_um=tv_sigma_um, random_seed=13
    )
    profiles_recon.append(recons)

# -----------------------------
# Confidence intervals (95%)
# -----------------------------
def ci_band(samples, alpha=0.05):
    lo = np.percentile(samples, 100*(alpha/2), axis=0)
    hi = np.percentile(samples, 100*(1 - alpha/2), axis=0)
    med = np.median(samples, axis=0)
    return lo, med, hi

ci_results = []
for recons in profiles_recon:
    lo, med, hi = ci_band(recons, alpha=0.05)
    ci_results.append((lo, med, hi))

# -----------------------------
# Statistical testing (basal window)
# Compare deep-region mean ε_r across realizations:
#   healthy vs psoriasis, healthy vs edema
# -----------------------------
def region_mean_per_realization(recons, z_um, z_min=50, z_max=120):
    m = []
    mask = (z_um >= z_min) & (z_um <= z_max)
    for r in recons:
        m.append(np.mean(r[mask]))
    return np.array(m)

deep_means = [region_mean_per_realization(recons, z_um, 50, 120) for recons in profiles_recon]
# healthy=0, psoriasis=1, edema=2
U_hp, p_hp = mannwhitneyu(deep_means[0], deep_means[1], alternative='two-sided')
U_he, p_he = mannwhitneyu(deep_means[0], deep_means[2], alternative='two-sided')

print("="*64)
print("Mann–Whitney U tests on deep/basal region (50–120 µm):")
print(f"Healthy vs Psoriasis: U={U_hp:.1f}, p={p_hp:.3e}")
print(f"Healthy vs Edema:     U={U_he:.1f}, p={p_he:.3e}")
print("="*64)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
for ax, case, eps_true, (lo, med, hi) in zip(axes, cases, profiles_true, ci_results):
    # True profile
    ax.plot(z_um, eps_true, color=case["color"], lw=2.8, label=f"True ({case['name']})")

    # Recon median
    ax.plot(z_um, med, color="k", lw=1.8, ls="--", label="Reconstructed (median)")

    # 95% CI band
    ax.fill_between(z_um, lo, hi, color="0.75", alpha=0.6, label="95% CI")

    # Draw SC thickness
    ax.axvline(case["d_sc"], color="gray", lw=1.5, ls=":", alpha=0.9)
    ax.text(case["d_sc"] + 1.5, eps_true.min()+0.1, r"$d_{\rm SC}$ = " + f"{case['d_sc']:.0f} µm",
            color="gray", fontsize=10, va="bottom")

    ax.set_title(case["name"])
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel(r"$\epsilon_r$ at 1 THz")
for ax in axes:
    ax.set_xlabel(r"Depth $z$ ($\mu$m)")
    ax.set_xlim(0, 200)
    ax.set_ylim(6.0, 9.4)

# Legend (one shared)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)

# Annotate p-values on the middle subplot area (above)
axes[1].text(0.5, 1.08,
             f"Mann–Whitney (deep region):  "
             f"Healthy vs Psoriasis p = {p_hp:.2e}   |   Healthy vs Edema p = {p_he:.2e}",
             transform=axes[1].transAxes, ha='center', va='bottom', fontsize=10)

fig.suptitle("Pathology Discrimination via Permittivity Profiles at 1 THz", y=1.15, fontsize=13)
plt.tight_layout()
plt.show()
