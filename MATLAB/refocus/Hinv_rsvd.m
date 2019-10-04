%H_INV_RSVD Computes Regularized SVD-Based Inverse of H Matrix
% Hinv = Hinv_rsvd(delays,f,apod,param)
% INPUTS:
%   delays = N x M matrix (N transmits, M elements) of delays (in samples)
%   f = Normalized Frequency (1/samples) Ranging from 0 to 1
%   apod = N x M matrix of apodizations
%   param = Regularization Parameter
% OUTPUTS:
%   Hinv = M x N matrix inverse of H 
function Hinv = Hinv_rsvd(delays,f,apod,param)
    % Forward Model Matrix
    H = H_model_matrix(delays,f,apod);
    % Compute SVD of H
    [U,S,V] = svd(H); 
    % Default Regularization Parameter Value
    if(isempty(param))
        param = 1e-3;
    end
    % Regularized Inversion Singular Value Matrix
    Sinv = S./(S.^2+(param*S(1))^2); % Regularize small values
    Sinv(S==0) = 0; % Only preserve the diagonal
    % Regularized Inversion of Model Matrix
    Hinv = V*Sinv'*U';
end