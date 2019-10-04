%H_MODEL_MATRIX Computes H Matrix Based on Delays, Frequency, and Apodization
% H = ForwardModel(delays,f,apod)
% INPUTS:
%   delays = N x M matrix (N transmits, M elements) of delays (in samples)
%   f = Normalized Frequency (1/samples) Ranging from 0 to 1
%   apod = N x M matrix of apodizations
% OUTPUTS:
%   H = N x M model matrix
function H = H_model_matrix(delays,f,apod)
    % Model Matrix with Delay, Frequency, and Apodization
    H = apod.*exp(-1j*2*pi*f*delays);
end

