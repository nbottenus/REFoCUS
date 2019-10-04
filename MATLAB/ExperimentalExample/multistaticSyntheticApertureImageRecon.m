clear
clc

% Setup path
addpath(genpath([pwd, '/../focusfun_MATLAB']));
addpath(genpath([pwd, '/../refocus']));

% Load Recovered Multistatic Channel Data
%load AdjointRecoveredMultistaticData.mat; 
load TikhonovRecoveredMultistaticData.mat; 

%% Reconstruct Multistatic Synthetic Aperture Image

% Points to Focus and Get Image At
dBrange = [-60, 0]; 
num_x = 150; num_z = 150; 
xlims = [-7e-3, 7e-3];
x_img = linspace(xlims(1), xlims(2), num_x);
zlims = [16e-3, 29e-3];
z_img = linspace(zlims(1), zlims(2), num_z);
c = 1460; % Sound Speed in Phantom

% Full Synthetic Aperture Image Reconstruction
[Z, Y, X] = meshgrid(z_img, 0, x_img);
foc_pts = [X(:), Y(:), Z(:)];
tic; txFocData = focus_fs(time, full_synth_data, ...
    foc_pts, rxAptPos, rxAptPos, 0, 0, c); toc;
txFocData = reshape(txFocData, ...
    [numel(z_img), numel(x_img), length(rxAptPos)]);
bModeImg = squeeze(sum(txFocData,3));

% Remove Low Frequency Artifact
fs = 1./mean(diff(time));
N = 5; Wn = (1e6)/fs;
[b, a] = butter(N, Wn, 'high');
bModeImg = filtfilt(b, a, bModeImg);

% Upsample B-Mode Image for Display Purposes
x_show = linspace(xlims(1), xlims(2), 1000);
z_show = linspace(zlims(1), zlims(2), 1000);
[X_show, Z_show] = meshgrid(x_show, z_show);
[X, Z] = meshgrid(x_img, z_img);
bModeImg_show = interp2(X, Z, abs(bModeImg), X_show, Z_show, 'spline');

% Show Reconstructed Image
imagesc(1000*x_img, 1000*z_img, ...
    real(20*log10(bModeImg_show/max(bModeImg_show(:)))), dBrange); 
axis image; xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('DAS Beamforming'); ylim(1000*(zlims+(2e-3)*[1,-1]));
colormap(gray); colorbar();