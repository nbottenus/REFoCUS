%H_INV_ADJOINT Computes Adjoint of H Matrix
% Hinv = Hinv_adjoint(delays,f,apod,param)
% INPUTS:
%   delays = N x M matrix (N transmits, M elements) of delays (in samples)
%   f = Normalized Frequency (1/samples) Ranging from 0 to 1
%   apod = N x M matrix of apodizations
%   param = function handle as a function of frequency (e.g. param = @(f) f)
% OUTPUTS:
%   Hinv = M x N matrix inverse of H 
function Hinv = Hinv_adjoint(delays,f,apod,param)
    % Forward Model Matrix
    H = H_model_matrix(delays,f,apod);
    % Adjoint (Conjugate Transpose)
    Hinv = H';
    % Default (No Ramp Filter)
    if(~isempty(param))
        Hinv = param(f)*Hinv;
    end
end
