import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

# ---- Hydration & dual-Debye (real-part proxy at 1 THz) ----
def phi_profile(z_um, phi_s=0.20, phi_b=0.70, dSC=15.0, w=5.0):
    return np.clip(phi_b + (phi_s - phi_b)*erfc((z_um - dSC)/w), 0.05, 0.95)

def eps_r_from_phi(phi):
    # Parameters consistent with manuscript
    eps_inf = 2.5
    Delta1, Delta2 = 74.0, 3.0
    f = 1e12; w = 2*np.pi*f
    tau1, tau2 = 8.3e-12, 0.3e-12
    # Real part of dual Debye
    term1 = Delta1*phi/(1+(w*tau1)**2)
    term2 = Delta2*(1-phi)/(1+(w*tau2)**2)
    return eps_inf + term1 + term2

# ---- Build three cases (depth only for profiles) ----
z = np.linspace(0, 200, 400)  # µm, fine axis for display

def make_profile(case):
    if case == 'healthy':
        phi = phi_profile(z, phi_s=0.20, phi_b=0.70, dSC=15, w=5)
    elif case == 'psoriasis':
        phi = phi_profile(z, phi_s=0.10, phi_b=0.70, dSC=30, w=5)
    elif case == 'edema':
        phi = phi_profile(z, phi_s=0.20, phi_b=0.85, dSC=15, w=5)
    else:
        raise ValueError
    return eps_r_from_phi(phi)

profiles_true = {
    'healthy'  : make_profile('healthy'),
    'psoriasis': make_profile('psoriasis'),
    'edema'    : make_profile('edema')
}

# ---- "Inverse reconstructions": TV-like denoising on noisy profiles ----
def tv_proxy_reconstruct(eps_true, snr_db=40, lam=1e-2, iters=60):
    np.random.seed()
    # add noise
    sig = np.linalg.norm(eps_true)/np.sqrt(eps_true.size)
    noise = (sig/10**(snr_db/20.0))*np.random.randn(eps_true.size)
    x = eps_true + noise
    # simple 1D ROF-like proximal gradient (TV proxy with Gaussian smoothing)
    u = x.copy()
    for _ in range(iters):
        u = (1-lam)*u + lam*gaussian_filter1d(u, sigma=1.0)
    return u

Nrep = 50
rec_stats = {}
for k, prof in profiles_true.items():
    rec = np.stack([tv_proxy_reconstruct(prof, snr_db=40, lam=0.15, iters=60)
                    for _ in range(Nrep)], axis=0)
    rec_stats[k] = {
        'mean': rec.mean(axis=0),
        'low' : np.percentile(rec, 2.5, axis=0),
        'high': np.percentile(rec,97.5, axis=0),
        'all' : rec
    }

# ---- Statistical discrimination (Mann-Whitney on basal permittivity) ----
def basal_metric(p):  # average over deep region
    return p[z>60].mean()

m_healthy  = np.array([basal_metric(r) for r in rec_stats['healthy']['all']])
m_psor     = np.array([basal_metric(r) for r in rec_stats['psoriasis']['all']])
m_edema    = np.array([basal_metric(r) for r in rec_stats['edema']['all']])
p_hp = mannwhitneyu(m_healthy, m_psor, alternative='two-sided').pvalue
p_he = mannwhitneyu(m_healthy, m_edema, alternative='two-sided').pvalue

# ---- Plot ----
plt.figure(figsize=(8.5,5.5))

def plot_case(idx, key, color, title):
    plt.subplot(1,3,idx)
    plt.fill_between(z, rec_stats[key]['low'], rec_stats[key]['high'],
                     color=color, alpha=0.2, label='95% CI')
    plt.plot(z, rec_stats[key]['mean'], color=color, lw=2.5, label='Reconstructed (mean)')
    plt.plot(z, profiles_true[key], 'k--', lw=1.5, label='True')
    plt.axvline(15 if key!='psoriasis' else 30, color='gray', ls=':', lw=1)
    plt.xlabel(r'Depth $z$ ($\mu$m)')
    if idx==1: plt.ylabel(r'$\langle\epsilon_r\rangle_x$')
    plt.title(title)
    plt.xlim(0,200); plt.ylim(6.0,9.2); plt.grid(alpha=0.3)
    if idx==1:
        plt.legend(loc='lower right', fontsize=8)

plot_case(1,'healthy'  ,'tab:blue',   '(a) Healthy')
plot_case(2,'psoriasis','tab:orange', '(b) Psoriasis (SC 30 μm)')
plot_case(3,'edema'    ,'tab:green',  r'(c) Edema ($\phi_{\mathrm{basal}}=0.85$)')

plt.suptitle(f'Pathology Discrimination  (Mann–Whitney: Healthy vs Psoriasis p={p_hp:.3e}, Healthy vs Edema p={p_he:.3e})')
plt.tight_layout(rect=[0,0,1,0.93])
plt.show()
