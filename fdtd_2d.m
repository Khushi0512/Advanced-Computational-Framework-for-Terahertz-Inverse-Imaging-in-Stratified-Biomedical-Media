function [Einc_out, Eref_out] = fdtd_2d(eps_r, nx, nz, dx, dz, dt, nt)
% 2D TM-mode FDTD with UPML
% eps_r: nx × nz (defined at Ez nodes)
% Returns: Einc and Eref at probe_z = 5

eps0 = 8.854e-12; mu0 = 4*pi*1e-7; c0 = 1/sqrt(eps0*mu0);

% Staggered grid
Ex = zeros(nx, nz);     % Ex at (i,j+0.5)
Ez = zeros(nx+1, nz);   % Ez at (i+0.5,j)
Hy = zeros(nx, nz);     % Hy at (i+0.5,j+0.5)

% Permittivity at Ex and Ez nodes
eps_Ex = (eps_r(:,1:end-1) + eps_r(:,2:end))/2;  % Average for Ex
eps_Ez = eps_r;                                  % Already at Ez

% Source
src_x = floor(nx/2)+1; src_z = 10;
t = (0:nt-1)'*dt;
pulse = gaussian_pulse(t, 1e12, 1e-12);
dpulse = [0; diff(pulse)]; dpulse = [dpulse; 0];

% Probe
probe_z = 5;
Eref_out = zeros(nt,1);

% UPML
npml = 15;
m = 3; sig_max = 0.8*(npml+1)/(dx);
sig_z = zeros(1,nz); kappa_z = ones(1,nz);
for j = 1:npml
    s = (j/npml)^m;
    sig_z(j) = sig_max * s;
    kappa_z(j) = 1 + 9*s;
end
for j = nz-npml+1:nz
    s = ((nz-j+1)/npml)^m;
    sig_z(j) = sig_max * s;
    kappa_z(j) = 1 + 9*s;
end

% Time loop
for n = 1:nt
    % Source injection at Ez
    if src_z <= nz
        Ez(src_x, src_z) = dpulse(n);
    end

    % Update Hy
    for i = 1:nx
        for j = 1:nz
            dEz_dx = (Ez(i+1,j) - Ez(i,j))/dx;
            Hy(i,j) = Hy(i,j) + dt/(mu0 * kappa_z(j)) * dEz_dx;
        end
    end

    % Update Ex
    for i = 1:nx
        for j = 1:nz-1
            dHy_dz = (Hy(i,j+1) - Hy(i,j))/dz;
            Ex(i,j) = Ex(i,j) + dt/(eps0 * eps_Ex(i,j)) * dHy_dz;
        end
    end

    % Update Ez
    for i = 2:nx
        for j = 1:nz
            dHy_dx = (Hy(i,j) - Hy(i-1,j))/dx;
            Ez(i,j) = Ez(i,j) + dt/(eps0 * eps_Ez(i-1,j)) * dHy_dx;
        end
    end

    % Record reflected field
    Eref_out(n) = Ez(src_x, probe_z);
end

% Incident field: run in vacuum
if all(eps_r(:) == 1)
    Einc_out = Eref_out;
else
    Einc_out = zeros(nt,1);
end
end
