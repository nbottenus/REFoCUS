function [kz, tx_kspace] = pwResp2kSpace(kx, f, tx_pwResp, z, c)

% Create K-Space Grid Pulsed-Wave Transmit
% Spatial Sampling for FFT Axis
dz = mean(diff(z)); nzFFT = numel(z); nxFFT = numel(kx);
% FFT Axis for Axial Spatial Frequency
kz = mod(fftshift((0:nzFFT-1)/(dz*nzFFT))+1/(2*dz), 1/dz)-1/(2*dz);

% Calculate K-Space Representation of Transmit Beam
apodPulse_kx_kz = zeros(nzFFT, nxFFT);
for i = 1:numel(kx)
    apodPulse_kx_kz(:,i) = interp1(abs(f), ...
        tx_pwResp(i,:), c*sqrt(kx(i)^2+kz.^2)', 'linear', 0);
end

% Introduce Dipole [cos(theta)] Directivity
[Kx, Kz] = meshgrid(kx, kz);
tx_kspace = apodPulse_kx_kz .* abs(cos(atan2(Kx, Kz)));
    
end

