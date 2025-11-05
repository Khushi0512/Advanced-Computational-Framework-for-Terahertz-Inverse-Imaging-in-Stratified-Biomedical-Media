import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.ndimage import gaussian_filter

nx, nz = 200, 200
x = np.arange(nx); z = np.arange(nz)
X, Z = np.meshgrid(x, z, indexing='ij')
z_um = Z.astype(float)

def phi_profile(z_um, phi_s=0.20, phi_b=0.70, dSC=15.0, w=5.0):
    return np.clip(phi_b + (phi_s - phi_b)*erfc((z_um - dSC)/w), 0.05, 0.95)

def eps_r_from_phi(phi):
    eps_inf = 2.5
    Delta1, Delta2 = 74.0, 3.0
    f = 1e12; w = 2*np.pi*f
    tau1, tau2 = 8.3e-12, 0.3e-12
    term1 = Delta1*phi/(1+(w*tau1)**2)
    term2 = Delta2*(1-phi)/(1+(w*tau2)**2)
    return eps_inf + term1 + term2

phi = phi_profile(z_um)
eps_true = eps_r_from_phi(phi)

def reconstruct_proxy(eps_true, snr_db=40, lam=0.2, iters=50):
    np.random.seed()
    sig = np.linalg.norm(eps_true)/np.sqrt(eps_true.size)
    noise = (sig/10**(snr_db/20.0))*np.random.randn(*eps_true.shape)
    u = eps_true + noise
    for _ in range(iters):
        u = (1-lam)*u + lam*gaussian_filter(u, sigma=1.0)
    return u

N = 100
recs = np.stack([reconstruct_proxy(eps_true, snr_db=40, lam=0.2, iters=50)
                 for _ in range(N)], axis=0)
sigma_map = recs.std(axis=0)

plt.figure(figsize=(7.6,4.6))
im = plt.imshow(sigma_map.T, origin='lower', cmap='turbo',
                extent=[0,200,0,200], vmin=0, vmax=min(0.25, sigma_map.max()))
plt.colorbar(im, label=r'$\sigma_{\epsilon_r}$')
plt.contour(eps_true.T, levels=[6.5,7.5,8.5], colors='w', linewidths=0.8, alpha=0.6,
            extent=[0,200,0,200], origin='lower')
plt.axhline(15, color='w', ls='--', lw=1, alpha=0.8)
plt.xlabel(r'$x$ ($\mu$m)'); plt.ylabel(r'$z$ ($\mu$m)')
plt.title(r'Uncertainty map: $\sigma_{\epsilon_r}(x,z)$ (100 reconstructions, SNR = 40 dB)')
plt.tight_layout()
plt.show()
