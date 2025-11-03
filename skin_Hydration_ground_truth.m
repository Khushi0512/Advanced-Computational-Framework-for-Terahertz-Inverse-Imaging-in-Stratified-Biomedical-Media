% fig2_hydration_profile.m
z = 0:0.1:30; z_m = z*1e-6;
phi_surface = 0.20; phi_basal = 0.70;
d_sc = 15e-6; w = 5e-6;
phi = phi_basal + (phi_surface - phi_basal) * erfc((z_m - d_sc)/w);

figure; plot(z, phi, 'k-', 'LineWidth', 2); grid on;
xlabel('Depth in Skin (\mum)'); ylabel('Water Volume Fraction \phi(z)');
title('Stratum Corneum Hydration Gradient (Ground Truth)');
xlim([0 30]); ylim([0.15 0.75]);
text(16, 0.65, '\phi_{basal} = 0.70', 'FontSize', 12);
text(16, 0.25, '\phi_{surface} = 0.20', 'FontSize', 12);
