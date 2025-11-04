function p = gaussian_pulse(t, fc, tau)
% Differentiated Gaussian pulse for broadband THz
t0 = 3*tau;
p = -sqrt(2/pi) * (t - t0)/tau .* exp(-((t - t0)/tau).^2) .* cos(2*pi*fc*(t - t0));
p = p / max(abs(p));
end
