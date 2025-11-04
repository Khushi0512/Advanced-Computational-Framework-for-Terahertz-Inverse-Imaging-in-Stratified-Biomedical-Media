import numpy as np
import matplotlib.pyplot as plt

# Frequency range (0.1 THz – 3 THz)
freq = np.linspace(0.1, 3, 500)  # THz

# Parameters (approximate from literature)
# Bound water has weaker frequency dependence; free water rises sharply with frequency.
alpha_bound = 20 + 50 * np.exp(-((freq - 0.5)/0.3)**2)  # cm^-1
alpha_free = 5 + 30 * (freq - 1)**2 / (1 + (freq - 1)**2)  # cm^-1

# Weighting factors (percent contribution)
w_bound = np.clip(0.82 - 0.4*(freq - 1), 0, 1)  # bound water dominates <1 THz
w_free = 1 - w_bound

# Combined effective absorption coefficient
alpha_total = w_bound * alpha_bound + w_free * alpha_free

# Simulated "hydrogel phantom" data (validation points)
freq_exp = np.array([0.3, 0.6, 1.0, 1.5, 2.0, 2.5])
alpha_exp = alpha_total[np.searchsorted(freq, freq_exp)] + np.random.normal(0, 2, len(freq_exp))

# --- Plot ---
plt.figure(figsize=(8,5))
plt.plot(freq, alpha_total, 'k-', linewidth=2, label='Model (Effective absorption)')
plt.plot(freq, alpha_bound, 'b--', alpha=0.6, label='Bound water contribution')
plt.plot(freq, alpha_free, 'r--', alpha=0.6, label='Free water contribution')
plt.scatter(freq_exp, alpha_exp, color='orange', edgecolor='k', zorder=5, label='Hydrogel phantom data')

# --- Styling ---
plt.title(r"Frequency-dependent absorption coefficient", fontsize=14)
plt.xlabel("Frequency (THz)")
plt.ylabel(r"Absorption Coefficient  $\alpha$ (cm$^{-1}$)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(0.25, 80, "Bound water dominates\n(< 1 THz, ~82%)", color='blue', fontsize=10)
plt.text(2.1, 45, "Free water dominates\n(> 2 THz)", color='red', fontsize=10)

plt.tight_layout()
plt.show()
