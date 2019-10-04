clear
clc

% Setup path
addpath(genpath([pwd, '/../focusfun_MATLAB']));

% Load File Saved by GenFullSynthData.m
load ../../Data/multistaticDataFieldII.mat; % Cyst and Lesions Phantom

% Setup Transmit Imaging Case Here
txAptPos = rxAptPos; % Using same array to transmit and recieve
tx_focDepth = 0.030; % Transmit Focus in [m]
theta = 0; % Transmit Angle [rad]
tx_dir = [sin(theta),0,cos(theta)]; % Each Transmit Beam is Straight Ahead
tx_origin_x = -0.00365:0.00005:0.00365; % Transmit Origin in [m]

% Transmit Apodization
[X_XDCR, TX_ORIGIN_X] = meshgrid(x, tx_origin_x);
rect = @(x) x>-1/2 & x<1/2;
sigma_rect = 0.008; % [m]
apod = rect( (X_XDCR-TX_ORIGIN_X)/sigma_rect );

% Get Different Rx Data for Each Tx 
rxdata_multiTx = zeros(numel(time), size(rxAptPos,1), numel(tx_origin_x));
parfor kk = 1:numel(tx_origin_x)
    rxdata_multiTx(:,:,kk) = ...
        focus_fs_to_TxBeam(time, scat, rxAptPos, txAptPos, ...
        [tx_origin_x(kk),0,0], tx_dir, tx_focDepth, apod(kk, :), 0, c);
    disp(['Completed Transmit ', num2str(kk), ...
        ' at ' num2str(tx_origin_x(kk)), ' m']);
end

% Save Focused Transmit Data to File
clearvars scat;
save phantomFocTxDataWalkingAperture.mat