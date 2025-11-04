function eps_r = skin_profile(phi_2d, freq)
% phi_2d: nx × nz water fraction
% freq: scalar or vector (Hz)

eps_inf = 2.5;
tau1 = 8.3e-12; tau2 = 0.3e-12;
delta1 = 74.0; delta2 = 3.0;

omega = 2*pi*freq;
delta_eff = delta1 * phi_2d + delta2 * (1 - phi_2d);

eps_r = eps_inf + delta_eff ./ (1 + (omega*tau1).^2) ...
    + 5*(1-phi_2d) ./ (1 + (omega*tau2).^2);
eps_r = real(eps_r);
eps_r = max(eps_r, 2.0);
end
