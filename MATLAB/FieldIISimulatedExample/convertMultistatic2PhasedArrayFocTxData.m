clear
clc

% Setup path
addpath(genpath([pwd, '/../focusfun_MATLAB']));

% Load File Saved by GenFullSynthData.m
load ../../Data/multistaticDataFieldII.mat; % Cyst and Lesions Phantom

% Setup Transmit Imaging Case Here
txAptPos = rxAptPos; % Using same array to transmit and recieve
tx_focDepth = 0.030; % Transmit Focus in [m]
theta = linspace(-pi, pi, 181)/4; % Transmit Angle [rad]
tx_origin = [0, 0, 0]; % Transmit Origin in [m]

% Get Different Rx Data for Each Tx 
rxdata_multiTx = zeros(numel(time), size(rxAptPos,1), numel(theta));
parfor kk = 1:numel(theta)
    % Each Transmit Beam is Straight Ahead
    tx_dir = [sin(theta(kk)), 0, cos(theta(kk))]; 
    rxdata_multiTx(:,:,kk) = ...
        focus_fs_to_TxBeam(time, scat, rxAptPos, txAptPos, ...
        tx_origin, tx_dir, tx_focDepth, ones(size(txAptPos,1),1), 0, c);
    disp(['Completed Transmit ', num2str(kk), ...
        ' at ' num2str(theta(kk)), ' Radians']);
end

% Save Focused Transmit Data to File
clearvars scat;
save phantomFocTxDataPhasedArray.mat