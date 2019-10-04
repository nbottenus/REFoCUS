function foc_data = focus_fs(t,signal,foc_pts,rxAptPos,varargin)
%
% FOCUS_FS - Focuses the RF data at desired locations
%
% The function interpolates the RF signals collected using the full synthetic sequence
% to focus the data at desired locations
%
% INPUTS:
% t                  - T x 1 time vector for samples of the input signal
% signal             - T x N x M matrix containing input RF data to be interpolated
% foc_pts            - P x 3 matrix with position of focal points [m]
% rxAptPos           - N x 3 matrix with positions of the Rx apertures (elements) [m]
% txAptPos           - M x 3 matrix with positions of the Tx apertures (elements) [m]
%                    - txAptPos = rxAptPos by default  
% dc_rx, dc_tx       - time offsets [s] for Tx and Rx; scalars, N (M) x 1 vectors, or P x N (M) matrix
% speed_of_sound     - speed of sounds [m/s]; default 1540 m/s
%
% OUTPUT:
% foc_data - P x N vector with interpolated (RF) data points
%

if nargin == 4
    txAptPos = rxAptPos;
    dc_rx = 0; dc_tx = 0;
    speed_of_sound = 1540;
elseif nargin == 5
    txAptPos = varargin{1};
    dc_rx = 0; dc_tx = 0;
    speed_of_sound = 1540;
elseif nargin == 6
    txAptPos = varargin{1};
    dc_rx = varargin{2}; dc_tx = dc_rx;
    speed_of_sound = 1540;
elseif nargin == 7
    txAptPos = varargin{1};
    dc_rx = varargin{2}; dc_tx = varargin{3};
    speed_of_sound = 1540;
elseif nargin == 8
    txAptPos = varargin{1};
    dc_rx = varargin{2}; dc_tx = varargin{3};
    speed_of_sound = varargin{4};
else
    error('Improper argument list');
end

% time from the focus to receive apertures (array elements)
rx_times = calc_times(foc_pts,rxAptPos,dc_rx,speed_of_sound);

% time from the transmit apertures (array elements) to focus
tx_times = calc_times(foc_pts,txAptPos,dc_tx,speed_of_sound);

% focused but not summed rf data
foc_data=zeros(size(foc_pts,1),size(rxAptPos,1));
for i=1:size(rx_times,2)
    for j=1:size(tx_times,2)
        % Simple 1D Interp From MATLAB
        foc_data(:,i)=foc_data(:,i)+interp1(t, signal(:,i,j), ...
            rx_times(:,i)+tx_times(:,j), 'linear', 0);
    end
end