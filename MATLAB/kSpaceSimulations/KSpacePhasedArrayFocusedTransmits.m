clear
clc

% Setup path
addpath(genpath([pwd, '/../focusfun_MATLAB']));
addpath(genpath([pwd, '/../refocus']));

% Methods of Recovery
%method = 'Adjoint';
method = 'RampFilter';
%method = 'Tikhonov';

%% Setup Wavefield and K-Space Coordinates

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
c = 1540; % m/usec
lambda = c/fc; 
elemSpace = 0.15e-3; % m
Nelem = 96; 
xpos = (-(Nelem-1)/2:(Nelem-1)/2)*elemSpace; 
apod = rectwin(Nelem);
steerAng = linspace(-pi, pi, 181)/4; % radians
focDepth = 0.030; % m

% Simulation Space and Time
Nx0 = 256; m = 2; n = 2; dov = 0.060; % m 
x = (-(Nx0*m-1)/2:(Nx0*m-1)/2)*(elemSpace/m); 
Nu1 = round(dov/(elemSpace/n)); 
z = ((0:Nu1-1))*elemSpace/n;
t = (0:0.05:2)*abs(focDepth)/c;

%% Ground-Truth Multistatic-Transmit Synthetic Aperture

% Calculate [K-Space, Wavefield, etc.] for Each Individual Transmit Element
multistatic_pwResp = zeros(numel(x), numel(f), Nelem); % Pulse-Wave Frequency Response
multistatic_kspace = zeros(numel(z), numel(x), Nelem); % K-Space Response
for elem_idx = 1:Nelem
    single_element = zeros(size(apod)); 
    single_element(elem_idx) = 1; % Single Element Apodization
    [kx, multistatic_pwResp(:,:,elem_idx)] = ... % Pulse-Wave Frequency Response
        pwResp(x, elemSpace, single_element, zeros(Nelem,1), P_f, f, c);
    [kz, multistatic_kspace(:,:,elem_idx)] = ... % K-Space Response
        pwResp2kSpace(kx, f, multistatic_pwResp(:,:,elem_idx), z, c);
end
[Kx, Kz] = meshgrid(kx, kz); % K-Space Grid
K = sqrt(Kx.^2 + Kz.^2); % Radius in K-Space

%% Transmit Pulse-Wave Frequency Response for Each Transmit Beam

% Pulse-Wave Frequency Response for Each Transmit Beam
tx_pwResp = zeros(numel(x), numel(f), numel(steerAng));
tx_delays = zeros(numel(steerAng), Nelem);
tx_apod = zeros(numel(steerAng), Nelem);
for steer_ang_idx = 1:numel(steerAng)
    % Calculating Transmit Delays for Each Transmit Beam
    if isinf(focDepth)
        tx_delays(steer_ang_idx, :) = xpos*sin(steerAng(steer_ang_idx))/c;
    else
        tx_delays(steer_ang_idx, :) = (sign(focDepth)*sqrt(xpos.^2+focDepth^2- ...
            2*focDepth*xpos*sin(steerAng(steer_ang_idx)))-focDepth)/c;
    end
    % Calculating Transmit Apodization for Each Transmit Beam
    tx_apod(steer_ang_idx, :) = apod;
    % Pulse-Wave Frequency Response for Each Transmit Beam
    [kx, tx_pwResp(:,:,steer_ang_idx)] = ...
        pwResp(x, elemSpace, apod, tx_delays(steer_ang_idx, :), P_f, f, c);
end

% Calculate K-Space Response For Each Transmit Beam
tx_kspace = zeros(numel(z), numel(x), numel(steerAng)); % K-Space Response
for steerAng_idx = 1:numel(steerAng)
    [~, tx_kspace(:,:,steerAng_idx)] = ... % K-Space Response
        pwResp2kSpace(kx, f, tx_pwResp(:,:,steerAng_idx), z, c);
end

% Reconstruct Transmit Wavefield for Transmit Beam
steerAng_idx = 121;
[~, ~, psf_t] = kspace2wavefield(kx, kz, ...
    (Kz>0).*tx_kspace(:,:,steerAng_idx), c, t);

% K-Space of a Single Transmit Beam
figure; imagesc(kx, kz, abs(tx_kspace(:,:,steerAng_idx))); 
axis image; axis xy; xlabel('lateral frequency [1/m]'); 
ylabel('axial frequency [1/m]'); colorbar;
title('K-Space of Selected Transmit Beam');

%% Simulate Multistatic Synthetic Aperture Recovery Techniques

% Decode Multistatic data Using REFoCUS
if strcmp(method, 'Adjoint')
    multistatic_recov_pwResp = ...
        multistatic_recov(kx, f, tx_pwResp, tx_apod, tx_delays, @Hinv_adjoint, @(f) 1);
elseif strcmp(method, 'RampFilter')
    multistatic_recov_pwResp = (elemSpace*mean(diff(steerAng))/c) * ...
        multistatic_recov(kx, f, tx_pwResp, tx_apod, tx_delays, @Hinv_adjoint, @(f) f);
elseif strcmp(method, 'Tikhonov')
    multistatic_recov_pwResp = ...
        multistatic_recov(kx, f, tx_pwResp, tx_apod, tx_delays, @Hinv_tikhonov, 1e-1);
end

% Calculate K-Space Responses For Each Recovered Element
multistatic_recov_kspace = zeros(numel(z), numel(x), Nelem); % K-Space Response
for elem_idx = 1:Nelem
    [~, multistatic_recov_kspace(:,:,elem_idx)] = ... % K-Space Response
        pwResp2kSpace(kx, f, multistatic_recov_pwResp(:,:,elem_idx), z, c);
end

%% K-Space and Wavefield for Single Element Transmits

% K-Space of the Adjoint-Based Transmit Response
figure; subplot(1,2,1); imagesc(kx, kz, mean(abs(multistatic_kspace), 3)); 
axis image; axis xy; xlabel('lateral frequency [1/m]'); 
ylabel('axial frequency [1/m]'); colorbar;
title('K-Space of True Single Element Response');

% K-Space of the Ramp-Filtered Adjoint Transmit Response
subplot(1,2,2); imagesc(kx, kz, mean(abs(multistatic_recov_kspace), 3)); 
axis image; axis xy; xlabel('lateral frequency [1/m]'); 
ylabel('axial frequency [1/m]'); colorbar;
title('K-Space of Recovered Single Element Response');

% Wavefield Due to Each Individual Transmit Element 
elem_idx = 48;
[~, ~, psf_t_recon] = kspace2wavefield(kx, kz, ...
    (Kz>0).*multistatic_recov_kspace(:,:,elem_idx), c, t);
[~, ~, psf_t_true] = kspace2wavefield(kx, kz, ...
    (Kz>0).*multistatic_kspace(:,:,elem_idx), c, t);

%% Plotting the Resulting Wavefield
maxpsf_t_recon = max(abs(psf_t_recon(~isinf(psf_t_recon) & ~isnan(psf_t_recon)))); 
maxpsf_t_true = max(abs(psf_t_true(~isinf(psf_t_true) & ~isnan(psf_t_true)))); 
maxpsf_t = max(abs(psf_t(~isinf(psf_t) & ~isnan(psf_t)))); 
figure; M = moviein(length(t)); kk = 1; 
while(1)
    subplot(1,3,1);
    imagesc(x,z,real(psf_t_true(:,:,kk)),0.1*maxpsf_t_true*[-1,1]);
    zoom on; axis equal; axis xy; axis image; colorbar;
    ylabel('z Axial Distance (mm)');
    xlabel('x Azimuthal Distance (mm)');
    title('True Single Element Wavefield');
    subplot(1,3,2); 
    imagesc(x,z,real(psf_t_recon(:,:,kk)),0.1*maxpsf_t_recon*[-1,1]);
    zoom on; axis equal; axis xy; axis image; colorbar;
    ylabel('z Axial Distance (mm)');
    xlabel('x Azimuthal Distance (mm)');
    title('Recovered Single Element Wavefield');
    subplot(1,3,3); 
    imagesc(x,z,real(psf_t(:,:,kk)),0.1*maxpsf_t*[-1,1]);
    zoom on; axis equal; axis xy; axis image; colorbar;
    ylabel('z Axial Distance (mm)');
    xlabel('x Azimuthal Distance (mm)');
    title('Selected Transmit Beam');
    M(kk) = getframe;
    if kk == length(t)
        kk = 1;
    else 
        kk = kk + 1;
    end
end