clear
clc

% Setup path
addpath(genpath([pwd, '/../focusfun_MATLAB']));
addpath(genpath([pwd, '/../refocus']));

% Pulse Definition
fc = 5.0e6; % Hz
fracBW = 0.7; 
fs = 20e6; % Hz

% Create Pulse in Both Time and Frequency Domain
Nf = 1024; t = (-Nf:Nf)/fs;  % (s) Time Vector centered about t=0
impResp = gauspuls(t,fc,fracBW); % Calculate Transmit Pulse
n = numel(impResp); P_f = fftshift(fft(impResp));
f = mod(fftshift((0:n-1)*fs/n)+fs/2, fs)-fs/2;
P_f = (f./(f+fc/10)).*abs(P_f); 
P_f = P_f(f>0); f = f(f>0); 

% Aperture Definition
c = 1540; % m/sec
lambda = c/fc; 
elemSpace = 0.15e-3; % m
Nelem = 128; 
apod = ones(Nelem,1); %apod = [zeros(63, 1); 1; zeros(64,1)];
steerAng = pi/12; % radians
focDepth = 0.020; % m
simDepth = 0.050; % m

% Simulation Space and Time
tFoc = simDepth/c; t = (0:0.01:1)*tFoc; m = 1; n = 2; 
Nx0 = 512; x = (-(Nx0-1)/2:(Nx0-1)/2)*(elemSpace/m); dov = 1; 
Nu1 = round(dov*c*max(t)/(elemSpace/n)); 
z = ((0:Nu1-1))*elemSpace/n;

% Transmit Delays Used in Beamforming
xpos = (-(Nelem-1)/2:(Nelem-1)/2)*elemSpace; 
if isinf(focDepth)
    delays = xpos*sin(steerAng)/c;
else
    delays = (sign(focDepth)*sqrt(xpos.^2+focDepth^2- ...
        2*focDepth*xpos*sin(steerAng))-focDepth)/c;
end

%% Transmit K-Space Representation and Pulse Wave Propagation (Direct Version)

% K-Space of the One-Way Transmit Response
[kx, tx_pwResp] = pwResp(x, elemSpace, apod, delays, P_f, f, c);
[kz, kspace] = pwResp2kSpace(kx, f, tx_pwResp, z, c);

% K-Space of the One-Way Transmit Response
figure; imagesc(kx, kz, abs(kspace)); 
axis image; axis xy; 
xlabel('lateral frequency [1/m]'); 
ylabel('axial frequency [1/m]');
title('K-Space of Transmit Response');

% Convert K-Space into Wavefield
kspace_downward = kspace; kspace_downward(kz<=0,:) = 0;
[~, ~, psf_t] = kspace2wavefield(kx, kz, kspace_downward, c, t);

% Plotting the Result
maxpsf_t = max(abs(psf_t(~isinf(psf_t) & ~isnan(psf_t)))); 
figure; M = moviein(length(t)); kk = 1; 
while(1)
    psf_tMag = (psf_t(:,:,kk));
    imagesc(x, z, real(psf_tMag/(maxpsf_t)), 0.1*[-1,1]);
    zoom on; axis equal; axis xy; axis image;
    ylabel('z Axial Distance (mm)');
    xlabel('x Azimuthal Distance (mm)');
    M(kk) = getframe;
    if kk == length(t)
        kk = 1;
    else 
        kk = kk + 1;
    end
end

