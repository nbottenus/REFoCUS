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

# Pulse Definition
fc = 5.0e6; # Hz
fracBW = 0.7;
fs = 20e6; # Hz

# Create Pulse in Both Time and Frequency Domain
Nf = 1024; t = np.arange(-Nf, Nf+1)/fs;  # (s) Time Vector centered about t=0
impResp = gausspulse(t, fc=fc, bw=fracBW); # Calculate Transmit Pulse
n = impResp.size; P_f = np.fft.fftshift(np.fft.fft(impResp));
f = np.mod(np.fft.fftshift(np.arange(n)*fs/n)+fs/2, fs)-fs/2;
P_f = (f/(f+fc/10))*np.abs(P_f); # Suppress DC component as much as possible
P_f = P_f[f>0]; f = f[f>0]; # Positive Frequencies

# Aperture Definition
c = 1540; # m/sec
Lambda = c/fc; # m
elemSpace = 0.15e-3; # m
Nelem = 128; # Number of Elements on Array
apod = np.ones(Nelem); #apod = [zeros(63, 1); 1; zeros(64,1)];
steerAng = np.pi/12; # radians
focDepth = 20e-3; # m
simDepth = 50e-3; # m

# Simulation Space and Time
tFoc = simDepth/c;
t = np.linspace(0,1,101)*tFoc;
m = 1; n = 2; Nx0 = 512; dov = 1;
x = (elemSpace/m)*np.arange(-(Nx0-1)/2, 1+(Nx0-1)/2);
Nu1 = np.round(dov*c*np.max(t)/(elemSpace/n));
z = (elemSpace/n)*np.arange(Nu1);

# Transmit Delays Used in Beamforming
xpos = elemSpace*np.arange(-(Nelem-1)/2, 1+(Nelem-1)/2);
if np.isinf(focDepth):
    delays = xpos*np.sin(steerAng)/c;
else:
    delays = (np.sign(focDepth)*np.sqrt(xpos**2+focDepth**2- \
        2*focDepth*xpos*np.sin(steerAng))-focDepth)/c;

## Transmit K-Space Representation and Pulse Wave Propagation (Direct Version)

# K-Space of the One-Way Transmit Response
kx, tx_pwResp = pwResp(x, elemSpace, apod, delays, P_f, f, c);
kz, kspace = pwResp2kSpace(kx, f, tx_pwResp, z, c);

# K-Space of the One-Way Transmit Response
plt.figure(); imagesc(kx, kz, np.abs(kspace), (0,np.max(np.abs(kspace))));
plt.xlabel('lateral frequency [1/m]'); plt.ylabel('axial frequency [1/m]');
plt.title('K-Space of Transmit Response'); plt.show();

# Convert K-Space into Wavefield
kspace_downward = kspace; kspace_downward[kz<=0,:] = 0;
xg, zg, psf_t = kspace2wavefield(kx, kz, kspace_downward, c, t);

# Plotting the Result
maxpsf_t = np.max(np.abs(psf_t[(~np.isinf(psf_t)) & (~np.isnan(psf_t))]));
plt.figure(); tpause = 1e-9; kk = 1; #M = moviein(length(t));
while True:
    imagesc(x, z, np.real(psf_t[:,:,kk]/maxpsf_t), (-0.1,0.1));
    plt.xlabel('x Azimuthal Distance (mm)');
    plt.ylabel('z Axial Distance (mm)');
    if kk == t.size-1:
        kk = 1;
    else:
        kk = kk + 1;
    plt.draw();
    plt.pause(tpause);
    plt.clf();
