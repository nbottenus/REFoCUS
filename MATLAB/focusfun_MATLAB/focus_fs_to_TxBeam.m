function foc_data = focus_fs_to_TxBeam(t, signal, rxAptPos, ...
    txAptPos, tx_center, tx_dir, tx_focDepth, tx_apod, dc_tx, speed_of_sound)
%
% FOCUS_FS - Focuses the RF data at desired locations
%
% The function interpolates the RF signals collected using the full synthetic sequence
% to focus the data at desired locations
%
% INPUTS:
% t                  - T x 1 time vector for samples of the input signal
% signal             - T x N x M matrix containing input RF data to be interpolated
% rxAptPos           - N x 3 matrix with positions of the Rx apertures (elements) [m]
% txAptPos           - M x 3 matrix with positions of the Tx apertures (elements) [m]
% tx_center          - 1 x 3 vector with the position of the center of the Tx aperture [m]
% tx_dir             - 1 x 3 matrix with direction of transmit beam
% tx_focDepth        - Depth of transmit focus along transmit direction [m] 
% tx_apod            - M x 1 vector of apodizations for transmit beam
% dc_tx              - time offsets [s] for Tx; scalars, M x 1 vectors, or T x M matrix
% speed_of_sound     - speed of sounds [m/s]; default 1540 m/s
%
% OUTPUT:
% foc_data - T x N vector with interpolated (RF) data points
%
% NOTE: this f-ion uses cubic spline interpolation (slower than linear interp in focus_fs_fast)
%

txAptPosRelToCtr = txAptPos - ones(size(txAptPos,1),1) * tx_center;
txFocRelToCtr = tx_focDepth * ones(size(txAptPos,1),1) * tx_dir/norm(tx_dir);
txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;

% positive value is time delay, negative is time advance
if isinf(tx_focDepth)
    tx_delay = ((-txAptPosRelToCtr)*(tx_dir'/norm(tx_dir)))/speed_of_sound;
else % Column Vector
    tx_delay = (sqrt(sum(txFocRelToCtr.^2, 2)) - sqrt(sum(txFocRelToAptPos.^2, 2)))/speed_of_sound;
end 
tx_delay = tx_delay + dc_tx; % Add time delay offsets

% transmit beamforming on full-synthetic aperture dataset: delayed-and-summed
foc_data = zeros(numel(t), size(rxAptPos,1));
for i = 1:size(rxAptPos,1)
    for j = 1:size(txAptPos,1)
        foc_data(:,i) = foc_data(:,i) + tx_apod(j) * ...
            interp1(t(:), signal(:,i,j), t(:)-tx_delay(j), 'linear', 0);
    end
end

end