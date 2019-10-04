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
xlims = [-10e-3, 10e-3];
x_img = linspace(xlims(1), xlims(2), num_x);
zlims = [20e-3, 40e-3];
z_img = linspace(zlims(1), zlims(2), num_z);
c = 1460; % Sound Speed in Phantom

% Full Synthetic Aperture Image Reconstruction
[Z, Y, X] = meshgrid(z_img, 0, x_img);
foc_pts = [X(:), Y(:), Z(:)];
tic; focData = focus_fs(time, full_synth_data, ...
    foc_pts, rxAptPos, rxAptPos, 0, 0, c); toc;
txFocData = reshape(focData, ...
    [numel(z_img), numel(x_img), length(rxAptPos)]);
bModeImg = squeeze(sum(txFocData,3));

% SLSC on Receive Data After Transmit Focusing Everywhere
numLags = length(rxAptPos)-1; % Number of Lags for SLSC
VCZcurve = zeros(numLags,1);
VCZ = @(focData, lag) squeeze( mean( mean( real( mean( ...
    (focData(:,:,1:end-lag).*conj(focData(:,:,lag+1:end))) ./ ...
    (abs(focData(:,:,1:end-lag)).*abs(focData(:,:,lag+1:end))), 3)), 1), 2));
for lag = 1:numLags
    VCZcurve(lag) = VCZ(txFocData, lag);
    disp(['SLSC Lag = ', num2str(lag)]);
end

% Plot VCZ Curves
figure; plot(1:numLags, VCZcurve, 'Linewidth', 2);
xlabel('lag'); ylabel('Coherence');