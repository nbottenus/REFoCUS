clear
clc

% Setup path
addpath(genpath([pwd, '/../focusfun_MATLAB']));
addpath(genpath([pwd, '/../refocus']));

% Load Channel Data from Walking-Aperture Tramsmits
load ../../Data/DATA_FocTx_20190323_205714.mat; 

% Load Focused Transmit Channel Data
rxdata_multiTx = rxdata_multiTx / max(rxdata_multiTx(:));
[nt, nRx, nTx] = size(rxdata_multiTx);
fs = 1./mean(diff(time));

% Decode Multistatic Data Using Tikhonov Regularization REFoCUS
rf_decoded = refocus_decode(rxdata_multiTx,tx_delays'*fs, ...
    'fun',@Hinv_tikhonov,'apod',apod,'param',1e-2);

% Hilbert Transform the Decoded Dataset
full_synth_data = reshape(hilbert(reshape(rf_decoded, ...
    [nt, nRx*nRx]), nt), [nt, nRx, nRx]);

% Passband Filter Channel Data
N = 10; % Filter Order
fs = 1./mean(diff(time));
Wn = [3,12]*(1e6)/(fs/2); % Pass Band
[b, a] = butter(N, Wn); % Filter
full_synth_data = filtfilt(b, a, full_synth_data);

% Save Recovered Multistatic Data
save('TikhonovRecoveredMultistaticData.mat', ...
    '-v7.3', 'time', 'rxAptPos', 'full_synth_data');