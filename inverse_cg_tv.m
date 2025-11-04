function [eps_rec, J_hist] = inverse_cg_tv(Eref_meas, Einc, eps_init, ...
    phi_2d, freq, nx, nz, dx, dz, dt, nt, max_iter)

eps_rec = eps_init;
J_hist = zeros(max_iter,1);
lambda_tv = 1e-4;

for k = 1:max_iter
    [~, Esim] = fdtd_2d(eps_rec, nx, nz, dx, dz, dt, nt);
    residual = Esim - Eref_meas;
    J_data = 0.5 * sum(residual.^2);
    J_tv = tv_norm(eps_rec);
    J_hist(k) = J_data + lambda_tv * J_tv;

    % Finite-difference gradient (sparse sampling)
    grad = zeros(nx,nz);
    delta = 1e-6;
    idx_i = round(linspace(20, nx-20, 6));
    idx_j = round(linspace(20, nz-20, 6));
    for ii = idx_i
        for jj = idx_j
            eps_pert = eps_rec; eps_pert(ii,jj) = eps_pert(ii,jj) + delta;
            [~, Epert] = fdtd_2d(eps_pert, nx, nz, dx, dz, dt, nt);
            grad(ii,jj) = sum((Epert - Esim) .* residual) / delta;
        end
    end
    grad = imgaussfilt(grad, 4);

    % TV gradient
    grad_tv = tv_gradient(eps_rec);
    grad_total = grad + lambda_tv * grad_tv;

    % CG update
    if k == 1
        d = -grad_total;
    else
        beta = max(0, sum(grad_total(:).*(grad_total(:)-grad_old(:))) / sum(grad_old(:).^2));
        d = -grad_total + beta * d;
    end
    grad_old = grad_total;

    % Line search
    alpha = 1e-3;
    for ls = 1:8
        eps_new = eps_rec + alpha * d;
        eps_new = max(eps_new, 2.0);
        J_new = cost_function(eps_new, Eref_meas, nx, nz, dx, dz, dt, nt) ...
            + lambda_tv * tv_norm(eps_new);
        if J_new < J_hist(k) || k==1
            break;
        end
        alpha = alpha * 0.5;
    end
    eps_rec = eps_new;

    fprintf('Iter %d | Cost: %.4e\n', k, J_new);
end
end

function J = cost_function(eps_r, Eref_meas, nx, nz, dx, dz, dt, nt)
[~, Esim] = fdtd_2d(eps_r, nx, nz, dx, dz, dt, nt);
J = 0.5 * sum((Esim - Eref_meas).^2);
end

function g = tv_gradient(u)
[ux, uz] = gradient(u);
mag = sqrt(ux.^2 + uz.^2 + 1e-12);
g = divergence(ux./mag, uz./mag);
end

function tv = tv_norm(u)
[ux, uz] = gradient(u);
tv = sum(sqrt(ux(:).^2 + uz(:).^2));
end

function [ux, uz] = gradient(u)
ux = [u(:,2:end) - u(:,1:end-1), zeros(size(u,1),1)];
uz = [u(2:end,:) - u(1:end-1,:); zeros(1,size(u,2))];
end

function div = divergence(px, pz)
divx = [px(:,1), px(:,2:end)-px(:,1:end-1), -px(:,end)];
divz = [pz(1,:); pz(2:end,:)-pz(1:end-1,:); -pz(end,:)];
div = divx(:,1:end-1) + divz(1:end-1,:);
end
