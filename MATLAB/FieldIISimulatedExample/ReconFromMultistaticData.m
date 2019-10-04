clear
clc

% Setup path
addpath(genpath([pwd, '/../focusfun_MATLAB']));

% Ground Truth Multistatic Channel Data
load ../../Data/multistaticDataFieldII.mat; % Cyst and Lesions Phantom

% Take Hilbert Transform of Field II Simulated Data
[nt, nRx, nTx] = size(scat);
scat_h = reshape(hilbert(reshape(scat, [nt, nRx*nTx])), [nt, nRx, nTx]);

% Points to Focus and Get Image At
dBrange = [-60, 0];
num_x = 150; num_z = 600; 
xlims = [-pitch*(no_elements-1)/4, pitch*(no_elements-1)/4];
x_img = linspace(xlims(1), xlims(2), num_x);
zlims = [5e-3, 3.5e-2];
z_img = linspace(zlims(1), zlims(2), num_z);

% Full Synthetic Aperture Image Reconstruction
[Z, Y, X] = meshgrid(z_img, 0, x_img);
foc_pts = [X(:), Y(:), Z(:)];
tic; txFocData = focus_fs(time, scat_h, foc_pts, rxAptPos, rxAptPos, 0, 0, c); toc;
txFocData = reshape(txFocData, [numel(z_img), numel(x_img), length(rxAptPos)]);
bModeImg = squeeze(sum(txFocData,3)); figure; subplot(1,2,1);
imagesc(1000*x_img, 1000*z_img, 20*log10(abs(bModeImg)/max(abs(bModeImg(:)))), dBrange); 
axis image; xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('B-Mode'); colormap(gray); colorbar();

% SLSC on Receive Data After Transmit Focusing Everywhere
numLags = nRx-1; % Number of Lags for SLSC
SLSCImg = zeros([size(bModeImg), numLags]);
SLSC = @(focData, lag) real( mean( ...
    (focData(:,:,1:end-lag).*conj(focData(:,:,lag+1:end))) ./ ...
    ( abs(focData(:,:,1:end-lag)).*abs(focData(:,:,lag+1:end)) ), 3) );
for lag = 1:numLags
    SLSCImg(:,:,lag) = SLSC(txFocData, lag);
    disp(['SLSC Lag = ', num2str(lag)]);
end
nlagsSLSC = 30; 
subplot(1,2,2); imagesc(1000*x_img, 1000*z_img, ...
    mean(SLSCImg(:,:,1:nlagsSLSC), 3), [0,1]); 
axis image; xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('SLSC'); colormap(gray); colorbar();

% VCZ Curves at Two Depths
VCZ1 = squeeze(mean(mean(SLSCImg(75:125,50:100,:),1),2));
VCZ2 = squeeze(mean(mean(SLSCImg(275:325,50:100,:),1),2));
figure; plot(VCZ1, 'Linewidth', 2); hold on; 
plot(VCZ2, 'Linewidth', 2); xlabel('lag'); 
ylabel('Coherence'); legend('10 mm depth', '20 mm depth');