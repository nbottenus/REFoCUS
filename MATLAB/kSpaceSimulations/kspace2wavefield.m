function [x, z, psf_t] = kspace2wavefield(kx, kz, kspace, c, t)

% Propagation of Pulse in Time Domain
[Kx, Kz, T] = meshgrid(kx, kz, t);
delayFactors = exp(-1i*2*pi*c*sqrt(Kx.^2+Kz.^2).*T); 

% Construction of Pulse-Wave Response vs Time
psf_kx_kz_t = repmat(kspace, [1, 1, numel(t)]) .* delayFactors;
nx = numel(kx); nz = numel(kz); psf_t = zeros(nz, nx, numel(t));
for t_idx = 1:numel(t)
    psf_t(:,:,t_idx) = ifft2(ifftshift(psf_kx_kz_t(:,:,t_idx)));
end

% Spatial Grid
dx = 1/(2*max(abs(kx))); dz = 1/(2*max(abs(kz)));
x = dx*((-(nx-1)/2):((nx-1)/2)); z = dz*(0:(nz-1));

end

