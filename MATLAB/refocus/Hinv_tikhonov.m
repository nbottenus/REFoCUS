%H_INV_TIKHONOV Computes Tikhonov Regularized Inverse of H Matrix
% Hinv = Hinv_tikhonov(delays,f,apod,param)
% INPUTS:
%   delays = N x M matrix (N transmits, M elements) of delays (in samples)
%   f = Normalized Frequency (1/samples) Ranging from 0 to 1
%   apod = N x M matrix of apodizations
%   param = Regularization Parameter
% OUTPUTS:
%   Hinv = M x N matrix inverse of H 
function Hinv = Hinv_tikhonov(delays,f,apod,param)
    % Forward Model Matrix
    H = H_model_matrix(delays,f,apod);
    % Default Regularization Parameter Value
    if(isempty(param))
        param = 1e-3;
    end
    % Regularization Relative to Maximum Singular Value
    smax = norm(H,2); N = size(H,2); 
    reg = param*smax*eye(N);
    % Tikhonov Regularized Inverse Matrix
    Hinv = (H'*H + reg'*reg)\H';
end