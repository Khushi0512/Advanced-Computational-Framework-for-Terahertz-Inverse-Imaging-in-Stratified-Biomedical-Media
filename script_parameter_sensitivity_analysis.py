import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.ndimage import gaussian_filter

nx, nz = 160, 160
Z = np.tile(np.linspace(0,200,nz),(nx,1))
def phi_profile(z_um, phi_s=0.20, phi_b=0.70, dSC=15.0, w=5.0):
    return np.clip(phi_b + (phi_s - phi_b)*erfc((z_um - dSC)/w), 0.05, 0.95)
def eps_r_from_phi(phi):
    eps_inf=2.5; D1, D2 = 74.0, 3.0
    f=1e12; w=2*np.pi*f; t1=8.3e-12; t2=0.3e-12
    return eps_inf + D1*phi/(1+(w*t1)**2) + D2*(1-phi)/(1+(w*t2)**2)

eps_true = eps_r_from_phi(phi_profile(Z))

def recon(eps_true, snr_db, lam, iters=40):
    sig = np.linalg.norm(eps_true)/np.sqrt(eps_true.size)
    noise = (sig/10**(snr_db/20))*np.random.randn(*eps_true.shape)
    u = eps_true + noise
    for _ in range(iters):
        u = (1-lam)*u + lam*gaussian_filter(u, sigma=1.0)
    return u

lams = np.logspace(-6, -2, 12)
snrs = np.linspace(20, 60, 11)
RMS = np.zeros((len(snrs), len(lams)))
for i, snr in enumerate(snrs):
    for j, lam in enumerate(lams):
        u = recon(eps_true, snr_db=snr, lam=lam)
        RMS[i,j] = np.sqrt(np.mean((u - eps_true)**2))

# ridge (argmin over columns)
ridge_j = RMS.argmin(axis=1)

plt.figure(figsize=(7.6,5.2))
im = plt.imshow(RMS, origin='lower', aspect='auto',
                extent=[np.log10(lams[0]), np.log10(lams[-1]), snrs[0], snrs[-1]],
                cmap='magma')
plt.colorbar(im, label='RMS error')
# highlight ridge
x_ridge = np.log10(lams[ridge_j])
plt.plot(x_ridge, snrs, 'w-', lw=2, label='Optimal ridge (L-curve)')
plt.xlabel(r'$\log_{10}\,\lambda_{\mathrm{TV}}$'); plt.ylabel('SNR (dB)')
plt.title('Parameter sensitivity: RMS error vs. TV weight and noise')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
