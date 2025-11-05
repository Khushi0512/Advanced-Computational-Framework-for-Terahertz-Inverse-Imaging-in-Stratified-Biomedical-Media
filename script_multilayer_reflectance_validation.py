import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from numpy.fft import rfft, rfftfreq

# ------------------ Physical constants (SI) ------------------
c0   = 2.99792458e8
mu0  = 4e-7*np.pi
eps0 = 1.0/(mu0*c0**2)

# ------------------ Three-layer skin phantom ----------------
d_SC   = 15e-6
d_epi  = 35e-6
d_derm = 150e-6
er_SC, er_epi, er_derm = 3.2, 4.5, 5.8
layers = [(er_SC, d_SC), (er_epi, d_epi), (er_derm, d_derm)]

# ------------------ Grid ------------------
dx   = 0.5e-6
pad_front = 100
pad_back  = 100
n_layers = int((d_SC + d_epi + d_derm)/dx)
Nz = pad_front + n_layers + pad_back
z  = np.arange(Nz)*dx

# ------------------ Time Discretization ------------------
dt = 0.99 * dx / c0
Nt = 16000
t  = np.arange(Nt)*dt

# ------------------ Source & Probe ------------------
src_index   = int(0.5 * pad_front)
probe_index = int(0.7 * pad_front)

fc  = 1.0e12
tau = 0.15e-12
t0  = 6*tau
src = -(2*(t - t0)/tau**2)*np.exp(-((t - t0)/tau)**2)*np.cos(2*np.pi*fc*(t - t0))

# ------------------ Build permittivity array ------------------
er = np.ones(Nz)
cursor = pad_front
n_SC   = int(round(d_SC/dx))
n_epi  = int(round(d_epi/dx))
n_derm = int(round(d_derm/dx))

er[cursor:cursor+n_SC]   = er_SC;   cursor += n_SC
er[cursor:cursor+n_epi]  = er_epi;  cursor += n_epi
er[cursor:cursor+n_derm] = er_derm

# ------------------ Mur ABC ------------------
mur_coef = (c0*dt - dx) / (c0*dt + dx)

# --------------------------------------------------------------
# ✅ FIXED FDTD FUNCTION — NO GLOBAL VARIABLES
# --------------------------------------------------------------
def run_fdtd(er_profile):
    """1D Yee FDTD with Mur absorbing boundaries."""
    
    # local fields (each run gets its own fresh arrays)
    Ex = np.zeros(Nz)
    Hy = np.zeros(Nz-1)

    rec = np.zeros(Nt)

    # Mur "previous" boundary values
    left_prev  = 0.0
    right_prev = 0.0

    for n in range(Nt):

        # --- Update Hy ---
        Hy += (dt/(mu0*dx)) * (Ex[1:] - Ex[:-1])

        # --- Hard source ---
        Ex[src_index] += src[n]

        # --- Update Ex interior ---
        curl = Hy[1:] - Hy[:-1]    # length Nz-2
        Ex[1:-1] += (dt/(eps0 * er_profile[1:-1] * dx)) * curl

        # --- Mur absorbing boundaries ---
        # left
        new_left = Ex[1] + mur_coef*(Ex[1] - left_prev)
        left_prev = Ex[0]
        Ex[0] = new_left

        # right
        new_right = Ex[-2] + mur_coef*(Ex[-2] - right_prev)
        right_prev = Ex[-1]
        Ex[-1] = new_right

        # --- Record ---
        rec[n] = Ex[probe_index]

    return rec

# ------------------ Run FDTD Twice ------------------
inc = run_fdtd(np.ones_like(er))   # vacuum
tot = run_fdtd(er)                 # layered medium
ref = tot - inc                    # reflected only

# ------------------ FFT ------------------
w = windows.hann(Nt)
Inc = rfft(inc*w)
Ref = rfft(ref*w)
freq = rfftfreq(Nt, dt)

mask = (freq>=0.1e12) & (freq<=3e12)
R_fdtd = np.abs(Ref[mask]/(Inc[mask]+1e-30))**2

# ------------------ Analytical TMM ------------------
def tmm_reflectance(freqs, layers):
    R = np.zeros_like(freqs)
    for idx, f in enumerate(freqs):
        w = 2*np.pi*f
        k0 = w/c0
        M = np.eye(2, dtype=complex)
        for er_k, d_k in layers:
            n_k = np.sqrt(er_k)
            dk  = k0*n_k*d_k
            P = np.array([[np.cos(dk), 1j*np.sin(dk)/n_k],
                          [1j*n_k*np.sin(dk), np.cos(dk)]])
            M = M @ P
        n0 = 1.0
        nN = np.sqrt(layers[-1][0])
        num = n0*M[0,0] + n0*nN*M[0,1] - (M[1,0] + nN*M[1,1])
        den = n0*M[0,0] + n0*nN*M[0,1] + (M[1,0] + nN*M[1,1])
        R[idx] = np.abs(num/den)**2
    return R

R_tmm = tmm_reflectance(freq[mask], layers)

# ------------------ Plot ------------------
plt.figure(figsize=(8,5))
plt.plot(freq[mask]*1e-12, R_fdtd, 'b-', lw=2, label='FDTD')
plt.plot(freq[mask]*1e-12, R_tmm, 'r--', lw=2, label='TMM')
plt.xlabel("Frequency (THz)")
plt.ylabel("Reflectance $R$")
plt.title("Skin Phantom: FDTD vs Analytical TMM")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
