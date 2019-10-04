function [kx, tx_pwResp] = pwResp(x, elemSpace, apod, delay, P_f, f, c)

% X-Coordinate of Apertures
Nelem = numel(apod); 
xpos = (-(Nelem-1)/2:(Nelem-1)/2)*elemSpace; 
apod_x = interp1(xpos, apod, x, 'spline', 0);
delayIdeal = interp1(xpos, delay, x, 'spline', 0);

% Complex Aperture Function
apod_x_cmpx = diag(apod_x)*exp(-1i*2*pi*delayIdeal'*f);

% Weight Complex Aperture Function by Pulse Fourier Transform
apodPulse = apod_x_cmpx .* repmat(P_f./(4*pi*f/c+eps), [numel(apod_x), 1]);

% Fourier Transform Along Lateral Axis
tx_pwResp = fftshift(fft(apodPulse, [], 1), 1); 

% Create K-Space Grid Pulsed-Wave Transmit
% Spatial Sampling for FFT Axis
dx = mean(diff(x)); nxFFT = numel(x); 
% FFT Axis for Axial Spatial Frequency
kx = mod(fftshift((0:nxFFT-1)/(dx*nxFFT))+1/(2*dx), 1/dx)-1/(2*dx);
    
end

