%H_INV_TSVD Computes Truncated SVD-Based Inverse of H Matrix
% Hinv = Hinv_tsvd(delays,f,apod,param)
% INPUTS:
%   delays = N x M matrix (N transmits, M elements) of delays (in samples)
%   f = Normalized Frequency (1/samples) Ranging from 0 to 1
%   apod = N x M matrix of apodizations
%   param = Truncation Parameter
% OUTPUTS:
%   Hinv = M x N matrix inverse of H 
function Hinv = Hinv_tsvd(delays,f,apod,param)
    % Forward Model Matrix
    H = H_model_matrix(delays,f,apod);
    % Compute SVD of H
    [U,S,V] = svd(H);
    % Default Truncation Parameter Value
    if(isempty(param))
        param = 1e-3;
    end
    % Truncate Singular Value Matrix
    Sinv = 1./S;
    Sinv(S<param*S(1)) = 0; % Truncate small values
    Sinv(S==0) = 0; % Only preserve the diagonal
    % Regularized Inversion of Model Matrix
    Hinv = V*Sinv'*U';
end