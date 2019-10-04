function fsa_recon_pwResp = multistatic_recov(kx, f, tx_pwResp, tx_apod, tx_delays, Hinv_fun, param)

% Pulse-Wave Frequency Response For Each Reconstructed Transmit Element
fsa_recon_pwResp = zeros(numel(kx), numel(f), size(tx_apod, 2));

% Frequency Domain Reconstruction
for f_idx = 1:numel(f)
    % Generate Forward Matrix Transformation
    Hinv = Hinv_fun(tx_delays, f(f_idx), tx_apod, param);
    % Reconstruct Full Synthetic Aperture Frequency-by-Frequency
    fsa_recon_pwResp(:,f_idx,:) = permute(Hinv * ...
        permute(squeeze(tx_pwResp(:,f_idx,:)), [2,1]), [2,1]);
end

end