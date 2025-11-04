% fig3_dielectric_debye.m
f = logspace(11, 13, 500); omega = 2*pi*f;
eps_inf = 2.5; tau1 = 8.3e-12; tau2 = 0.3e-12;
delta1 = 74.0; delta2 = 3.0;

phi = [0.2 0.7]; eps_r = zeros(length(f),2);
for i = 1:2
    delta_eff = delta1*phi(i) + delta2*(1-phi(i));
    eps_r(:,i) = eps_inf + delta_eff./(1 + 1i*omega*tau1) ...
        + 5*(1-phi(i))./(1 + 1i*omega*tau2);
end

figure('Position',[100,100,900,500]);
subplot(1,2,1);
plot(f*1e-12, real(eps_r), 'LineWidth', 1.8); grid on; legend('\phi=0.20','\phi=0.70');
xlabel('Frequency (THz)'); ylabel('Re\{\epsilon_r\}'); title('Real Part');
xlim([0.1 3]); ylim([2 12]);

subplot(1,2,2);
plot(f*1e-12, imag(eps_r), 'LineWidth', 1.8); grid on; legend('\phi=0.20','\phi=0.70');
xlabel('Frequency (THz)'); ylabel('Im\{\epsilon_r\}'); title('Imaginary Part (Absorption)');
xlim([0.1 3]); ylim([0 8]);
