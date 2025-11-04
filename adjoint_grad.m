function grad = adjoint_grad(Eref_meas, Einc, eps_r, nx, nz, dx, dz, dt, nt, freq)
% Compute dJ/d(eps_r) via adjoint method
Esim = fdtd_2d(eps_r, nx, nz, dx, dz, dt, nt); Esim = Esim(:);
Eref_meas = Eref_meas(:); Einc = Einc(:);
residual = Esim - Eref_meas;

% Sensitivity: dE/d(eps) via perturbation (finite difference)
delta = 1e-6;
grad = zeros(nx,nz);
for i = 1:nx
    for j = 1:nz
        eps_pert = eps_r;
        eps_pert(i,j) = eps_pert(i,j) + delta;
        Epert = fdtd_2d(eps_pert, nx, nz, dx, dz, dt, nt);
        grad(i,j) = sum((Epert(:) - Esim) .* residual) / delta;
    end
end
end
