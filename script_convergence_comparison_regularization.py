import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.ndimage import gaussian_filter, laplace

nx, nz = 200, 200
Z = np.tile(np.linspace(0,200,nz),(nx,1))
def phi_profile(z_um, phi_s=0.20, phi_b=0.70, dSC=15.0, w=5.0):
    return np.clip(phi_b + (phi_s - phi_b)*erfc((z_um - dSC)/w), 0.05, 0.95)
def eps_r_from_phi(phi):
    eps_inf=2.5; D1, D2 = 74.0, 3.0
    f=1e12; w=2*np.pi*f; t1=8.3e-12; t2=0.3e-12
    return eps_inf + D1*phi/(1+(w*t1)**2) + D2*(1-phi)/(1+(w*t2)**2)

truth = eps_r_from_phi(phi_profile(Z))
# “measured” = truth + noise
np.random.seed(0)
sigma = np.linalg.norm(truth)/np.sqrt(truth.size)
noise = (sigma/10**(40/20))*np.random.randn(*truth.shape)
g = truth + noise

def solve_no_reg(g, it=30, alpha=0.6):
    u = np.zeros_like(g); J=[]
    for k in range(it):
        grad = u - g
        u -= alpha*grad
        J.append(0.5*np.sum((u-g)**2))
    return u, np.array(J)

def solve_tikhonov(g, lam=1e-2, it=60, alpha=0.6):
    u = np.zeros_like(g); J=[]
    for k in range(it):
        grad = (u - g) + lam*laplace(u)
        u -= alpha*grad
        J.append(0.5*np.sum((u-g)**2) + 0.5*lam*np.sum(np.gradient(u,axis=0)[0]**2))
    return u, np.array(J)

def solve_tv_proxy(g, lam=0.15, it=60):
    u = np.zeros_like(g); J=[]
    for k in range(it):
        data = 0.5*np.sum((u-g)**2)
        u = (1-lam)*u + lam*gaussian_filter(u, sigma=1.1)
        J.append(data)
    return u, np.array(J)

u0, J0 = solve_no_reg(g, it=30, alpha=0.6)
u2, J2 = solve_tikhonov(g, lam=5e-3, it=60, alpha=0.5)
u1, J1 = solve_tv_proxy(g, lam=0.18, it=60)

def rms(a,b): return np.sqrt(np.mean((a-b)**2))
r0, r2, r1 = rms(u0,truth), rms(u2,truth), rms(u1,truth)

# Depth-averaged
z = np.linspace(0,200,nz)
def depth_avg(u): return u.mean(axis=0)

plt.figure(figsize=(9.2,4.6))

plt.subplot(1,2,1)
plt.semilogy(J0, 'r--', label=f'No reg (diverges)')
plt.semilogy(J2, 'C1-', label=f'Tikhonov (RMS={r2:.3f})')
plt.semilogy(J1, 'C0-', label=f'TV proxy (RMS={r1:.3f})')
plt.xlabel('Iteration'); plt.ylabel('Cost (log)')
plt.title('Convergence comparison')
plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.plot(z, depth_avg(truth), 'k', lw=2.5, label='True')
plt.plot(z, depth_avg(u0), 'r--', lw=1.2, label='No reg')
plt.plot(z, depth_avg(u2), 'C1-', lw=1.8, label='Tikhonov')
plt.plot(z, depth_avg(u1), 'C0-', lw=1.8, label='TV proxy')
plt.axvline(15, color='gray', ls=':', lw=1)
plt.xlabel(r'Depth $z$ ($\mu$m)'); plt.ylabel(r'$\langle\epsilon_r\rangle_x$')
plt.title('Recovered depth profiles')
plt.xlim(0,200); plt.ylim(6.0,9.2); plt.grid(alpha=0.3); plt.legend()

plt.tight_layout()
plt.show()
