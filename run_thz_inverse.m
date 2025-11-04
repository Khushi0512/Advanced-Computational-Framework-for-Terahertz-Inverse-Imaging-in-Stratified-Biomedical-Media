% run_thz_inverse.m
% Fully working: 2D FDTD + Inverse Reconstruction
clear; close all; clc;

%% Grid & Physics
nx = 256; nz = 256;
dx = 1e-6; dz = 1-6;
c0 = 3e8;
dt = 0.99 * dx / (c0 * sqrt(2));
nt = 1024;

%% Hydration Profile (Ground Truth)
phi_surface = 0.20; phi_basal = 0.70;
d_sc = 15e-6; w_gradient = 5e-6;
z = (0:nz-1)*dz;
phi_true = phi_basal + (phi_surface - phi_basal) * erfc((z - d_sc)/w_gradient);
phi_2d = repmat(phi_true', nx, 1);  % nx × nz

%% Dielectric Model (at 1 THz)
freq = 1e12;
eps_r = skin_profile(phi_2d, freq);  % nx × nz

%% === INCIDENT FIELD (Free Space) ===
eps_vacuum = ones(nx, nz);
[Einc, ~] = fdtd_2d(eps_vacuum, nx, nz, dx, dz, dt, nt);

%% === TOTAL FIELD (With Skin) ===
[~, Eref] = fdtd_2d(eps_r, nx, nz, dx, dz, dt, nt);

%% Add Noise
SNR = 40;
noise = randn(size(Eref)) * std(Eref) / 10^(SNR/20);
Eref_meas = Eref + noise;

%% Inverse Reconstruction
eps_init = 4.0 * ones(nx, nz);
[eps_rec, J_hist] = inverse_cg_tv(Eref_meas, Einc, eps_init, ...
    phi_2d, freq, nx, nz, dx, dz, dt, nt, 25);

%% Plot
plot_results(z*1e6, phi_true, eps_rec, Eref, Eref_meas, J_hist);

fprintf('Done. Final cost: %.4e\n', J_hist(end));
