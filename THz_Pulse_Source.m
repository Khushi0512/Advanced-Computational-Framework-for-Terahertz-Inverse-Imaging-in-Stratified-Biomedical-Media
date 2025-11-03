% fig1_thz_pulse.m
t = linspace(-5e-12, 10e-12, 2000);
fc = 1e12; tau = 1e-12;
pulse = gaussian_pulse(t, fc, tau);

% FFT
dt = t(2)-t(1); N = length(t);
f = (0:N-1)/(N*dt); f = f - f(end/2)*(f>max(f)/2);
P = abs(fftshift(fft(pulse)));

figure('Position',[100,100,900,400]);
subplot(1,2,1);
plot(t*1e12, pulse, 'LineWidth', 1.8); grid on;
xlabel('Time (ps)'); ylabel('E-field (a.u.)');
title('Differentiated Gaussian THz Pulse (1 THz Center)');
xlim([-3 6]); ylim([-1.1 1.1]);

subplot(1,2,2);
plot(f*1e-12, P/max(P), 'LineWidth', 1.8); grid on;
xlabel('Frequency (THz)'); ylabel('Normalized Spectrum');
title('THz Pulse Spectrum'); xlim([0 3]); ylim([0 1.1]);
