# Setting up all folders we can import from by adding them to python path
import sys, os, pdb
curr_path = os.getcwd();
sys.path.append(curr_path+'/..');

# Importing stuff from all folders in python path
import numpy as np
from focusfun import *
from refocus import *
from KSpaceFunctions import *

# TESTING CODE FOR FOCUS_DATA Below
import scipy.io as sio
from scipy.signal import hilbert, gausspulse
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# Methods of Recovery
method = 'Adjoint';
#method = 'RampFilter';
#method = 'Tikhonov';

# Pulse Definition
fc = 5.0e6; # Hz
fracBW = 0.7;
fs = 20e6; # Hz

# Create Pulse in Both Time and Frequency Domain
Nf = 1024; t = np.arange(-Nf,Nf+1)/fs; # (s) Time Vector centered about t=0
impResp = gausspulse(t, fc=fc, bw=fracBW); # Calculate Transmit Pulse
n = impResp.size; P_f = np.fft.fftshift(np.fft.fft(impResp));
f = np.mod(np.fft.fftshift(np.arange(n)*fs/n)+fs/2,fs)-fs/2;
P_f = (f/(f+fc/10))*np.abs(P_f);
P_f = P_f[f>0]; f = f[f>0];

# Aperture Definition
c = 1540; # m/usec
LAMBDA = c/fc;
elemSpace = 0.15e-3; # m
Nelem = 96;
xpos = np.arange(-(Nelem-1)/2, 1+(Nelem-1)/2)*elemSpace;
apod = np.ones(Nelem);
steerAng = np.linspace(-np.pi, np.pi, 181)/4; # radians
focDepth = 0.030; # m

# Simulation Space and Time
Nx0 = 256; m = 2; n = 2; dov = 0.060; # m
x = np.arange(-(Nx0*m-1)/2, 1+(Nx0*m-1)/2)*(elemSpace/m);
Nu1 = np.round(dov/(elemSpace/n));
z = (np.arange(Nu1))*elemSpace/n;
t = np.arange(0,2,0.05)*np.abs(focDepth)/c;

## Ground-Truth Multistatic-Transmit Synthetic Aperture

# Calculate [K-Space, Wavefield, etc.] for Each Individual Transmit Element
multistatic_pwResp = np.zeros((x.size, f.size, Nelem), dtype=np.complex); # Pulse-Wave Frequency Response
multistatic_kspace = np.zeros((z.size, x.size, Nelem), dtype=np.complex); # K-Space Response
for elem_idx in np.arange(Nelem):
    single_element = np.zeros(apod.shape);
    single_element[elem_idx] = 1; # Single Element Apodization
    # Pulse-Wave Frequency Response
    kx, multistatic_pwResp[:,:,elem_idx] = \
        pwResp(x, elemSpace, single_element, np.zeros(Nelem), P_f, f, c);
    # K-Space Response
    kz, multistatic_kspace[:,:,elem_idx] = \
        pwResp2kSpace(kx, f, multistatic_pwResp[:,:,elem_idx], z, c);
Kx, Kz = np.meshgrid(kx, kz); # K-Space Grid
K = np.sqrt(Kx**2 + Kz**2); # Radius in K-Space

## Transmit Pulse-Wave Frequency Response for Each Transmit Beam

# Pulse-Wave Frequency Response for Each Transmit Beam
tx_pwResp = np.zeros((x.size, f.size, steerAng.size), dtype=np.complex);
tx_delays = np.zeros((steerAng.size, Nelem), dtype=np.complex);
tx_apod = np.zeros((steerAng.size, Nelem), dtype=np.complex);
for steer_ang_idx in np.arange(steerAng.size):
    # Calculating Transmit Delays for Each Transmit Beam
    if np.isinf(focDepth):
        tx_delays[steer_ang_idx, :] = xpos*np.sin(steerAng[steer_ang_idx])/c;
    else:
        tx_delays[steer_ang_idx, :] = (np.sign(focDepth)*np.sqrt(xpos**2+focDepth**2 - \
            2*focDepth*xpos*np.sin(steerAng[steer_ang_idx]))-focDepth)/c;
    # Calculating Transmit Apodization for Each Transmit Beam
    tx_apod[steer_ang_idx, :] = apod;
    # Pulse-Wave Frequency Response for Each Transmit Beam
    kx, tx_pwResp[:,:,steer_ang_idx] = \
        pwResp(x, elemSpace, apod, tx_delays[steer_ang_idx,:], P_f, f, c);


# Calculate K-Space Response For Each Transmit Beam
tx_kspace = np.zeros((z.size, x.size, steerAng.size), dtype=np.complex); # K-Space Response
for steerAng_idx in np.arange(steerAng.size):
    _, tx_kspace[:,:,steerAng_idx] = \
        pwResp2kSpace(kx, f, tx_pwResp[:,:,steerAng_idx], z, c);

# Reconstruct Transmit Wavefield for Transmit Beam
steerAng_idx = 121;
_, _, psf_t = kspace2wavefield(kx, kz, (Kz>0)*tx_kspace[:,:,steerAng_idx], c, t);

# K-Space of a Single Transmit Beam
plt.figure(); imagesc(kx, kz, np.abs(tx_kspace[:,:,steerAng_idx]), \
    (0, np.max(np.abs(tx_kspace[:,:,steerAng_idx]))) );
plt.xlabel('lateral frequency [1/m]');
plt.ylabel('axial frequency [1/m]');
plt.title('K-Space of Selected Transmit Beam');
plt.show();

## Simulate Multistatic Synthetic Aperture Recovery Techniques

# Decode Multistatic data Using REFoCUS
if method == 'Adjoint':
    multistatic_recov_pwResp = \
        multistatic_recov(kx, f, tx_pwResp, tx_apod, tx_delays, Hinv_adjoint, lambda f: 1);
elif method == 'RampFilter':
    multistatic_recov_pwResp = (elemSpace*np.mean(np.diff(steerAng))/c) * \
        multistatic_recov(kx, f, tx_pwResp, tx_apod, tx_delays, Hinv_adjoint, lambda f: f);
elif method == 'Tikhonov':
    multistatic_recov_pwResp = \
        multistatic_recov(kx, f, tx_pwResp, tx_apod, tx_delays, Hinv_tikhonov, 1e-1);

# Calculate K-Space Responses For Each Recovered Element
multistatic_recov_kspace = np.zeros((z.size, x.size, Nelem), dtype=np.complex); # K-Space Response
for elem_idx in np.arange(Nelem): # K-Space Response
    _, multistatic_recov_kspace[:,:,elem_idx] = \
        pwResp2kSpace(kx, f, multistatic_recov_pwResp[:,:,elem_idx], z, c);

## K-Space and Wavefield for Single Element Transmits

# K-Space of the Adjoint-Based Transmit Response
plt.figure(); plt.subplot(1,2,1);
imagesc(kx, kz, np.mean(np.abs(multistatic_kspace), axis=2), \
    (0,np.max(np.mean(np.abs(multistatic_kspace), axis=2))) );
plt.xlabel('lateral frequency [1/m]');
plt.ylabel('axial frequency [1/m]');
plt.title('K-Space of True Single Element Response');

# K-Space of the Ramp-Filtered Adjoint Transmit Response
plt.subplot(1,2,2);
imagesc(kx, kz, np.mean(np.abs(multistatic_recov_kspace), axis=2), \
    (0,np.max(np.mean(np.abs(multistatic_recov_kspace), axis=2))) );
plt.xlabel('lateral frequency [1/m]');
plt.ylabel('axial frequency [1/m]');
plt.title('K-Space of Recovered Single Element Response');
plt.show();

# Wavefield Due to Each Individual Transmit Element
elem_idx = 48;
_, _, psf_t_recon = kspace2wavefield(kx, kz, \
    (Kz>0)*multistatic_recov_kspace[:,:,elem_idx], c, t);
_, _, psf_t_true = kspace2wavefield(kx, kz, \
    (Kz>0)*multistatic_kspace[:,:,elem_idx], c, t);

## Plotting the Resulting Wavefield
maxpsf_t_recon = np.max(np.abs(psf_t_recon[~np.isinf(psf_t_recon) & ~np.isnan(psf_t_recon)]));
maxpsf_t_true = np.max(np.abs(psf_t_true[~np.isinf(psf_t_true) & ~np.isnan(psf_t_true)]));
maxpsf_t = np.max(np.abs(psf_t[~np.isinf(psf_t) & ~np.isnan(psf_t)]));
plt.figure(); tpause = 1e-9; kk = 1;
while True:
    plt.subplot(1,3,1);
    imagesc(x,z,np.real(psf_t_true[:,:,kk]),0.1*maxpsf_t_true*np.array([-1,1]));
    plt.ylabel('z Axial Distance (mm)');
    plt.xlabel('x Azimuthal Distance (mm)');
    plt.title('True Single Element Wavefield');
    plt.subplot(1,3,2);
    imagesc(x,z,np.real(psf_t_recon[:,:,kk]),0.1*maxpsf_t_recon*np.array([-1,1]));
    plt.ylabel('z Axial Distance (mm)');
    plt.xlabel('x Azimuthal Distance (mm)');
    plt.title('Recovered Single Element Wavefield');
    plt.subplot(1,3,3);
    imagesc(x,z,np.real(psf_t[:,:,kk]),0.1*maxpsf_t*np.array([-1,1]));
    plt.ylabel('z Axial Distance (mm)');
    plt.xlabel('x Azimuthal Distance (mm)');
    plt.title('Selected Transmit Beam');
    if kk == t.size-1:
        kk = 1;
    else:
        kk = kk + 1;
    plt.draw();
    plt.pause(tpause);
    plt.clf();
