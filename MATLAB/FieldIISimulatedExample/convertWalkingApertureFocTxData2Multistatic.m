clear
clc

% Setup path
addpath(genpath([pwd, '/../focusfun_MATLAB']));
addpath(genpath([pwd, '/../refocus']));

% Load Channel Data from Walking-Aperture Tramsmits
load phantomFocTxDataWalkingAperture.mat; % Cyst and Lesions Phantom

% Method of Recovery
%method = 'Adjoint';
method = 'Tikhonov';

% Load Focused Transmit Channel Data
rxdata_multiTx = rxdata_multiTx / max(rxdata_multiTx(:));
[nt, nRx, nTx] = size(rxdata_multiTx);

% Assemble Focal Delays For Each Transmit Beam
tx_delays = zeros(size(txAptPos, 1), numel(tx_origin_x));
speed_of_sound = c;
for kk = 1:numel(tx_origin_x)
    % Calculate All Geometric Distances
    tx_center = [tx_origin_x(kk),0,0];
    txAptPosRelToCtr = txAptPos - ones(size(txAptPos,1),1) * tx_center;
    txFocRelToCtr = tx_focDepth * ones(size(txAptPos,1),1) * tx_dir/norm(tx_dir);
    txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;
    % Positive Value is Time Delay, Negative is Time Advance
    if isinf(tx_focDepth)
        tx_delay = ((-txAptPosRelToCtr)*(tx_dir'/norm(tx_dir)))/speed_of_sound;
    else
        tx_delay = (sqrt(sum(txFocRelToCtr.^2, 2)) - ...
            sqrt(sum(txFocRelToAptPos.^2, 2)))/speed_of_sound;
    end % Column Vector
    tx_delays(:, kk) = tx_delay;
end

% Decode Multistatic Data Using REFoCUS
if strcmp(method, 'Adjoint')
    rf_decoded = refocus_decode(rxdata_multiTx,tx_delays'*fs, ...
        'fun',@Hinv_adjoint,'apod',apod,'param',@(f) 1);
elseif strcmp(method, 'Tikhonov')
    rf_decoded = refocus_decode(rxdata_multiTx,tx_delays'*fs, ...
        'fun',@Hinv_tikhonov,'apod',apod,'param',1e-2);
end

% Hilbert Transform the Decoded Dataset
full_synth_data = reshape(hilbert(reshape(rf_decoded, ...
    [nt, nRx*nRx]), nt), [nt, nRx, nRx]);

%% Compare Recovered and Ground-Truth Multistatic Channel Data

% Ground Truth Multistatic Channel Data
load('../../Data/multistaticDataFieldII.mat', 'scat');

% Show Time-Gain-Compensated Channel Data
show_tx_channel = 1;
[~, tgc, ~] = meshgrid(1:no_elements, time, 1:no_elements); 
gnd_truth = scat .* tgc; 
gnd_truth = gnd_truth/max(gnd_truth(:));
recon = real(full_synth_data) .* tgc;
std_dev = std(gnd_truth(:));

% Normalize for Plotting
recon = recon * (gnd_truth(:)'*recon(:))/(recon(:)'*recon(:));

% Compare Recovered Channel Data to Ground Truth
figure; subplot(1,2,1); imagesc(x, time, squeeze(gnd_truth(:,:,show_tx_channel)))
xlabel('Lateral [m]'); ylabel('Time (s)'); 
title('True Channel Data'); caxis([-std_dev, std_dev]);
subplot(1,2,2); imagesc(x, time, squeeze(recon(:,:,show_tx_channel)) );
xlabel('Lateral [m]'); ylabel('Time (s)'); 
title('Reconstructed Channel Data'); caxis([-std_dev, std_dev]);

% Show Single Channel
show_rx_channel = 48;
figure; plot(time, squeeze(gnd_truth(:,show_rx_channel,show_tx_channel)), 'r', ...
    time, squeeze(recon(:,show_rx_channel,show_tx_channel)), 'b', 'Linewidth', 2); 
xlabel('Time (s)'); title('Single Channel Traces');
legend('True Channel Data', 'Reconstructed Channel Data'); 

% Time-Domain Reconstruction Error
cov_recon_true = cov(recon(:), gnd_truth(:));
corr_recon_gnd_truth = sqrt((cov_recon_true(1,2)*cov_recon_true(2,1)) / ...
    (cov_recon_true(1,1)*cov_recon_true(2,2)));
disp(['Correlation of Recovered Multistatic Dataset with Ground Truth is: ', ...
    num2str(corr_recon_gnd_truth)]);

%% Reconstruct Multistatic Synthetic Aperture Image

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
tic; txFocData = focus_fs(time, full_synth_data, ...
    foc_pts, rxAptPos, rxAptPos, 0, 0, c); toc;
txFocData = reshape(txFocData, ...
    [numel(z_img), numel(x_img), length(rxAptPos)]);
bModeImg = squeeze(sum(txFocData,3)); figure; subplot(1,2,1);
imagesc(1000*x_img, 1000*z_img, ...
    20*log10(abs(bModeImg)/max(abs(bModeImg(:)))), dBrange); 
axis image; xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('DAS Beamforming'); colormap(gray); colorbar();

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