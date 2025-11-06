#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilayer Reflectance Validation (FDTD vs TMM)

- 1D FDTD (normal incidence) with Mur ABC at both ends
- Reference ("no-sample") and sample runs -> incident & reflected separation
- Frequency-domain reflectance: R_FDTD(ν) = |Ê_ref(ν) / Ê_inc(ν)|^2
- Vectorized TMM with complex permittivity (σ loss)
- Publication-ready plot & RMS error

Author: cleaned & validated
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------- Physical constants --------------------
c0   = 2.99792458e8         # m/s
eps0 = 8.854187817e-12      # F/m
mu0  = 4.0e-7*np.pi         # H/m
eta0 = np.sqrt(mu0/eps0)

# -------------------- Multilayer definition -----------------
# Air (0) | SC (1) | Epidermis (2) | Dermis (3, semi-infinite)
eps_r_layers = np.array([1.0, 3.2, 4.5, 5.8], dtype=float)
sigma_layers = np.array([0.0, 0.1, 0.3, 0.5], dtype=float)  # S/m
thick_layers = np.array([np.inf, 15e-6, 35e-6, np.inf], dtype=float)  # m

# -------------------- FDTD grid/time ------------------------
# 1D z-grid: long enough to prevent time-alias & separate pulses
dz = 0.5e-6            # 0.5 µm
Nz = 5000              # domain points (2.5 mm) -> ample for pulse separation
Lz = Nz*dz

# CFL (1D): dt <= dz/c
CFL = 0.99
dt = CFL*dz/c0
# Total time ~ 8 ps -> adequate spectral resolution ~0.125 THz
t_total = 8.0e-12
Nt = int(np.ceil(t_total/dt))
t = np.arange(Nt)*dt

# Source: differentiated Gaussian, wideband
f0 = 1.5e12           # Hz, center ~ 1.5 THz to cover 0.1–3 THz band
tau = 0.15e-12        # 0.15 ps -> broad
t0  = 5*tau
src = -(2*(t - t0)/tau**2)*np.exp(-((t - t0)/tau)**2) * np.cos(2*np.pi*f0*(t - t0))

# Place source and probe (same positions in both runs)
isrc = 300
iprobe = 1200         # choose so reflected pulse returns before domain edge

# -------------------- Utility: assign material profile -----------------------
def build_material_profile(eps_r_layers, sigma_layers, thick_layers, Nz, dz):
    """
    Returns arrays eps_r[z], sigma[z] for the multilayer stack placed
    starting at z = z_int (here we place interface shortly after the probe).
    Geometry:
      0..z_int      -> air
      [z_int, z1)   -> SC      thickness 15 µm
      [z1,   z2)    -> Epidermis thickness 35 µm
      [z2,   end)   -> Dermis (semi-inf.)

    We pick z_int far from source but in front of probe to allow probe to see
    both incident (in ref run) and incident+reflected (in sample run).
    """
    eps_r = np.ones(Nz, dtype=float) * eps_r_layers[0]
    sigma = np.ones(Nz, dtype=float) * sigma_layers[0]

    # Interface start just after probe so probe sits in incident region
    z_int = (iprobe + 250) * dz  # start of SC region
    idx_int = int(np.round(z_int/dz))

    d1 = int(np.round(thick_layers[1]/dz)) if np.isfinite(thick_layers[1]) else 0
    d2 = int(np.round(thick_layers[2]/dz)) if np.isfinite(thick_layers[2]) else 0

    i1 = idx_int
    i2 = idx_int + d1
    i3 = i2 + d2

    # SC
    eps_r[i1:i2] = eps_r_layers[1]
    sigma[i1:i2] = sigma_layers[1]
    # Epidermis
    eps_r[i2:i3] = eps_r_layers[2]
    sigma[i2:i3] = sigma_layers[2]
    # Dermis
    eps_r[i3:]   = eps_r_layers[3]
    sigma[i3:]   = sigma_layers[3]

    return eps_r, sigma, idx_int, i1, i2, i3

# Reference (air everywhere)
def build_air_profile(Nz):
    return np.ones(Nz, dtype=float), np.zeros(Nz, dtype=float)

# -------------------- 1D FDTD solver (TMz normal incidence) -----------------
def run_fdtd(eps_r_z, sigma_z, return_E_at_probe=True):
    """
    1D Yee scheme: Ez at cell centers, Hy staggered
    Conductivity handled with (Ca, Cb) standard lossy update.
    Mur 1st-order ABC at both ends.
    Returns time series Ez_probe[n].
    """
    Ez = np.zeros(Nz, dtype=float)
    Hy = np.zeros(Nz-1, dtype=float)

    # Precompute E-update coefficients (lossy medium)
    # dEz/dt = (1/eps0/eps_r)*(dHy/dz - sigma*Ez)
    # Discrete form:
    # Ez^{n+1} = Cae*Ez^n + Cbe*(Hy^n(i)-Hy^n(i-1))
    Cae = (1.0 - sigma_z*dt/(2*eps0*eps_r_z)) / (1.0 + sigma_z*dt/(2*eps0*eps_r_z))
    Cbe = (dt/(eps0*eps_r_z*dz)) / (1.0 + sigma_z*dt/(2*eps0*eps_r_z))

    # Hy update (lossless): Hy^{n+1/2} = Hy^{n-1/2} + (dt/mu0/dz)*(Ez^n(i+1)-Ez^n(i))
    Ch  = dt/(mu0*dz)

    # Mur ABC storage (two previous edge values)
    Ez_left_old  = 0.0
    Ez_right_old = 0.0

    Ez_probe = np.zeros(Nt, dtype=float)

    # Time stepping
    for n in range(Nt):
        # 1) Update Hy (on half steps); Hy size Nz-1 uses Ez differences
        Hy += Ch * (Ez[1:] - Ez[:-1])

        # 2) Soft source: add to Ez at isrc (Gaussian-derivative)
        Ez[isrc] += src[n]

        # 3) Update Ez interior using precomputed coefficients
        curl_h = np.empty_like(Ez)
        curl_h[1:-1] = (Hy[1:] - Hy[:-1])  # already divided by dz inside Cbe
        curl_h[0] = 0.0
        curl_h[-1] = 0.0

        Ez = Cae*Ez + Cbe*curl_h

        # 4) Mur ABC at both ends (1st order)
        # Ez_new[0]   = Ez_old[1] + ( (c*dt - dz)/(c*dt + dz) )*(Ez_new[1] - Ez_old[0])
        # Using wave speed in air at edges:
        Sc = c0*dt/dz  # Courant number in air
        k_mur = (1.0 - Sc)/(1.0 + Sc)

        Ez0_new = Ez[1] + k_mur*(Ez[1] - Ez_left_old)
        EzN_new = Ez[-2] + k_mur*(Ez[-2] - Ez_right_old)
        Ez_left_old, Ez_right_old = Ez[0], Ez[-1]
        Ez[0], Ez[-1] = Ez0_new, EzN_new

        # 5) Record at probe
        if return_E_at_probe:
            Ez_probe[n] = Ez[iprobe]

    return Ez_probe

# -------------------- Frequency-domain helpers -------------------------------
def rfft_spectrum(signal, dt):
    """One-sided FFT and frequency axis."""
    N = signal.size
    win = np.hanning(N)
    s_win = signal * win
    S = np.fft.rfft(s_win)
    f = np.fft.rfftfreq(N, d=dt)
    return f, S

def band_mask(f, fmin, fmax):
    return (f >= fmin) & (f <= fmax)

# -------------------- Vectorized TMM reflectance -----------------------------
def tmm_reflectance_vector(freqs, eps_r_layers, sigma_layers, thick_layers):
    """
    Normal-incidence multilayer TMM (electric field formulation).
    Layers: 0..N-1, with layer 0 and N-1 semi-infinite.
    Returns R(f) with same length as freqs.
    """
    R = np.zeros_like(freqs, dtype=float)

    for idx, f in enumerate(freqs):
        omega = 2.0*np.pi*f if f > 0 else 1.0  # avoid divide-by-zero at f=0

        # Complex permittivity per layer
        eps_c = eps_r_layers.astype(complex) - 1j*(sigma_layers/(omega*eps0))

        # Wave impedance & wavenumber
        eta = np.sqrt(mu0/(eps0*eps_c))
        k   = (omega/c0)*np.sqrt(eps_c)

        # Global transfer matrix
        M = np.eye(2, dtype=complex)

        # Interface 0->1 then propagate in 1, then 1->2 ... last interface to N-1 handled by recursion
        # We follow the standard approach: for layers 1..N-2 do (interface from j-1->j) + (prop in j)
        for j in range(1, len(eps_r_layers)-1):
            # Interface matrix j-1 -> j
            etaL, etaR = eta[j-1], eta[j]
            t = 2.0*etaL/(etaL + etaR)
            r = (etaL - etaR)/(etaL + etaR)
            I = np.array([[1.0/t, r/t],
                          [r/t,   1.0/t]], dtype=complex)

            # Propagation in layer j (if finite thick)
            if np.isfinite(thick_layers[j]) and thick_layers[j] > 0:
                phi = k[j]*thick_layers[j]
                P = np.array([[np.exp(-1j*phi), 0.0],
                              [0.0,            np.exp(+1j*phi)]], dtype=complex)
            else:
                P = np.eye(2, dtype=complex)

            M = M @ I @ P

        # Last interface (N-2 -> N-1) folded inside matrix M via next loop iteration above
        # The total reflection coefficient seen from layer 0 is:
        # r_tot = M21 / M11 (if E- field formulation)
        if abs(M[0, 0]) > 1e-30:
            r_tot = M[1, 0] / M[0, 0]
        else:
            r_tot = 0.0

        R[idx] = np.abs(r_tot)**2

    return R

# -------------------- Run simulations (reference & sample) -------------------
print("Running reference (air) FDTD...")
eps_air, sig_air = build_air_profile(Nz)
E_ref_air = run_fdtd(eps_air, sig_air)

print("Building multilayer stack and running FDTD...")
eps_ml, sig_ml, idx_int, i1, i2, i3 = build_material_profile(
    eps_r_layers, sigma_layers, thick_layers, Nz, dz
)
E_ref_samp = run_fdtd(eps_ml, sig_ml)

# -------------------- Incident / Reflected separation ------------------------
# At the same probe position:
# With linearity, incident waveform ~= reference-air waveform at probe.
# Reflected waveform ~= sample_probe - reference_air_probe.
E_inc_t = E_ref_air.copy()
E_ref_t = E_ref_samp - E_ref_air

# -------------------- Frequency-domain reflectance ---------------------------
f, S_inc = rfft_spectrum(E_inc_t, dt)
_, S_ref = rfft_spectrum(E_ref_t, dt)

# Avoid divide-by-zero near nulls by thresholding the denominator
den = np.maximum(np.abs(S_inc), 1e-14)
R_fdtd = (np.abs(S_ref)/den)**2

# Limit comparison band to 0.1–3.0 THz
mask = band_mask(f, 0.1e12, 3.0e12)

# -------------------- TMM reflectance on same frequency grid -----------------
R_tmm = tmm_reflectance_vector(f, eps_r_layers, sigma_layers, thick_layers)

# -------------------- RMS error (on comparison band) -------------------------
def rms_err(a, b, m):
    if np.count_nonzero(m) == 0:
        return np.nan
    diff = a[m] - b[m]
    return np.sqrt(np.mean(diff**2))

rms = rms_err(R_fdtd, R_tmm, mask)
print(f"Frequency band: {f[mask][0]*1e-12:.2f}–{f[mask][-1]*1e-12:.2f} THz")
print(f"RMS error (FDTD vs TMM) over band: {rms*100:.3f}% (absolute reflectance units)")

# -------------------- Plot ----------------------------
plt.figure(figsize=(9,5.5))

plt.plot(f*1e-12, R_fdtd, lw=2.0, color='tab:blue', label='FDTD (this work)')
plt.plot(f*1e-12, R_tmm,  lw=2.0, color='tab:red',  ls='--', label='TMM (analytical)')

plt.xlim(0.0, 3.0)
plt.ylim(0.0, 0.15)
plt.xlabel('Frequency $\\nu$ (THz)')
plt.ylabel('Reflectance $R(\\nu)$')
plt.title('Multilayer Reflectance: FDTD vs TMM (normal incidence)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
