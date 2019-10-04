import numpy as np
import pdb

def pwResp(x, elemSpace, apod, delay, P_f, f, c):
    """ kx, tx_pwResp = pwResp(x, elemSpace, apod, delay, P_f, f, c)

    PWRESP - Pulsed-Wave Response as a Function of Lateral Wavenumber and Frequency

    This function computes transmit wavefield in terms of lateral wavenumber k_x
    and frequency f based on provided apodization and delay profile.

    INPUTS:
    x                 - 1 x M array of lateral positions x on computational grid [m]
    elemSpace         - spacing or pitch between transducer elements [m]
    apod              - 1 x K array of transmit apodization on transducer array
    delay             - 1 x K array of transmit delays [s] in same transmission event
    P_f               - 1 x N array for transmit pulse spectrum
    f                 - 1 x N array for frequencies in transmit pulse
    c                 - speed of sounds [m/s]; default 1540 m/s

    OUTPUTS:
    kx                - 1 x M array of lateral spatial frequencies [1/m]
    tx_pwResp         - M x N array for pulsed-wave response """

    # X-Coordinate of Apertures
    Nelem = apod.size;
    xpos = elemSpace*np.arange(-(Nelem-1)/2, 1+(Nelem-1)/2);
    apod_x = np.interp(x, xpos, apod, left=0, right=0);
    delayIdeal = np.interp(x, xpos, delay, left=0, right=0);

    # Complex Aperture Function
    apod_x_cmpx = np.dot(np.diag(apod_x), np.exp(-1j*2*np.pi*np.outer(delayIdeal,f)));

    # Weight Complex Aperture Function by Pulse Fourier Transform
    apodPulse = apod_x_cmpx * np.tile(P_f/(4*np.pi*f/c+np.finfo(np.float32).eps), (apod_x.size, 1));

    # Fourier Transform Along Lateral Axis
    tx_pwResp = np.fft.fftshift(np.fft.fft(apodPulse, axis=0), axes=0);

    # Create K-Space Grid Pulsed-Wave Transmit
    # Spatial Sampling for FFT Axis
    dx = np.mean(np.diff(x)); nxFFT = x.size;
    # FFT Axis for Axial Spatial Frequency
    kx = np.mod(np.fft.fftshift(np.arange(nxFFT)/(dx*nxFFT))+1/(2*dx), 1/dx)-1/(2*dx);
    return kx, tx_pwResp;

def pwResp2kSpace(kx, f, tx_pwResp, z, c):
    """ kz, tx_kspace = pwResp2kSpace(kx, f, tx_pwResp, z, c)

    PWRESP2KSPACE - Convert Pulsed-Wave Response to K-Space Representation

    This function converts transmit pulsed-wave response
    (in terms of lateral wavenumber k_x and frequency f)
    into the k-space for the transmitted wavefield
    (lateral and axial wavenumbers k_x and k_z).

    INPUTS:
    kx                - 1 x M array of lateral spatial frequencies [1/m]
    f                 - 1 x N array for frequencies in transmit pulse
    tx_pwResp         - M x N array for pulsed-wave response
    z                 - 1 x P array of depths [m]
    c                 - speed of sounds [m/s] (default 1540 m/s)

    OUTPUTS:
    kz                - 1 x P array of axial spatial frequencies [1/m]
    tx_kspace         - P x M array of k-space for transmit wavefield """

    # Create K-Space Grid Pulsed-Wave Transmit
    # Spatial Sampling for FFT Axis
    dz = np.mean(np.diff(z)); nzFFT = z.size; nxFFT = kx.size;
    # FFT Axis for Axial Spatial Frequency
    kz = np.mod(np.fft.fftshift(np.arange(nzFFT)/(dz*nzFFT))+1/(2*dz), 1/dz)-1/(2*dz);

    # Calculate K-Space Representation of Transmit Beam
    apodPulse_kx_kz = np.zeros((nzFFT, nxFFT), dtype=np.complex);
    for i in np.arange(kx.size):
        apodPulse_kx_kz[:,i] = np.interp(c*np.sqrt(kx[i]**2+kz**2), \
            np.abs(f), tx_pwResp[i,:], left=0, right=0);

    # Introduce Dipole [cos(theta)] Directivity
    Kx, Kz = np.meshgrid(kx, kz);
    tx_kspace = apodPulse_kx_kz * np.abs(np.cos(np.arctan2(Kx, Kz)));
    return kz, tx_kspace;

def kspace2wavefield(kx, kz, kspace, c, t):
    """ x, z, psf_t = kspace2wavefield(kx, kz, kspace, c, t)

    KSPACE2WAVEFIELD - Convert K-Space Representation to Physical Wavefield

    This function k-space for the transmitted wavefield
    (lateral and axial wavenumbers k_x and k_z) into physical wavefield
    (as a function of lateral and axial grid positions x and z).

    INPUTS:
    kx                - 1 x M array of lateral spatial frequencies [1/m]
    kz                - 1 x P array of axial spatial frequencies [1/m]
    kspace            - P x M array of k-space for transmit wavefield
    c                 - speed of sounds [m/s] (default 1540 m/s)
    t                 - 1 x T array of times [s]

    OUTPUTS:
    x                 - 1 x M array of lateral positions on computational grid [m]
    z                 - 1 x P array of axial positions on computational grid [m]
    psf_t             - P x M x T array of k-space for transmit wavefield """

    # Propagation of Pulse in Time Domain
    Kx, Kz, T = np.meshgrid(kx, kz, t);
    delayFactors = np.exp(-1j*2*np.pi*c*np.sqrt(Kx**2+Kz**2)*T);

    # Construction of Pulse-Wave Response vs Time
    psf_kx_kz_t = np.tile(kspace[:,:,np.newaxis], (1, 1, t.size)) * delayFactors;
    nx = kx.size; nz = kz.size; psf_t = np.zeros((nz, nx, t.size), dtype=np.complex);
    for t_idx in np.arange(t.size):
        psf_t[:,:,t_idx] = np.fft.ifft2(np.fft.ifftshift(psf_kx_kz_t[:,:,t_idx], axes=(0,1)));

    # Spatial Grid
    dx = 1/(2*np.max(np.abs(kx))); dz = 1/(2*np.max(np.abs(kz)));
    x = dx*np.arange((-(nx-1)/2),((nx-1)/2)+1); z = dz*np.arange(nz);
    return x, z, psf_t;

def multistatic_recov(kx, f, tx_pwResp, tx_apod, tx_delays, Hinv_fun, param):
    # Pulse-Wave Frequency Response For Each Reconstructed Transmit Element
    fsa_recon_pwResp = np.zeros((kx.size, f.size, tx_apod.shape[1]), dtype=np.complex);

    # Frequency Domain Reconstruction
    for f_idx in np.arange(f.size):
        # Generate Forward Matrix Transformation
        Hinv = Hinv_fun(tx_delays, f[f_idx], tx_apod, param);
        # Reconstruct Full Synthetic Aperture Frequency-by-Frequency
        fsa_recon_pwResp[:,f_idx,:] = np.transpose(np.dot(Hinv, \
            np.transpose(np.squeeze(tx_pwResp[:,f_idx,:]), axes=(1,0))), axes=(1,0));

    return fsa_recon_pwResp;
