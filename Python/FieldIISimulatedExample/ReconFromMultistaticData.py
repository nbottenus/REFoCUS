 # Setting up all folders we can import from by adding them to python path
import sys, os, pdb
curr_path = os.getcwd();
sys.path.append(curr_path+'/..');

# Importing stuff from all folders in python path
import numpy as np
from focusfun import *

# TESTING CODE FOR FOCUS_DATA Below
import scipy.io as sio
from scipy.signal import hilbert
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# Ground Truth Multistatic Channel Data
data_in = loadmat_hdf5('../../Data/multistaticDataFieldII.mat'); # Cyst and Lesions Phantom
time = data_in['time'][0];
scat = data_in['scat'];
scat_h = hilbert(scat, axis = 0);
rxAptPos = data_in['rxAptPos'];
pitch = np.mean(np.diff(rxAptPos[:,0]));
no_elements = rxAptPos.shape[0];
c = data_in['c'][0][0];

# Take Hilbert Transform of Field II Simulated Data
nt, nRx, nTx = scat.shape;

# Points to Focus and Get Image At
dBrange = np.array([-60, 0]);
num_x = 150; num_z = 600;
xlims = np.array([-pitch*(no_elements-1)/4, pitch*(no_elements-1)/4]);
x_img = np.linspace(xlims[0], xlims[1], num_x);
zlims = np.array([5e-3, 3.5e-2]);
z_img = np.linspace(zlims[0], zlims[1], num_z);

# Full Synthetic Aperture Image Reconstruction
(Z, Y, X) = np.meshgrid(z_img, 0, x_img);
foc_pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()));
txFocData = focus_fs(time, scat_h, foc_pts, rxAptPos, rxAptPos, 0, 0, c);
txFocData = txFocData.reshape(z_img.size, x_img.size, rxAptPos.shape[0]);
bModeImg = np.squeeze(np.sum(txFocData, axis = 2));
plt.figure(); plt.subplot(1,2,1);
imagesc(1000*x_img, 1000*z_img, \
    20*np.log10(np.abs(bModeImg)/np.max(np.abs(bModeImg[:]))), dBrange);
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('DAS Beamforming');

# SLSC on Receive Data After Transmit Focusing Everywhere
numLags = nRx-1; # Number of Lags for SLSC
SLSCImg = np.zeros((num_z, num_x, numLags));
SLSC = lambda focData, lag: np.real( np.mean( \
    ( focData[:,:,:-lag] * np.conj(focData[:,:,lag:]) ) / \
    ( np.abs(focData[:,:,:-lag]) * np.abs(focData[:,:,lag:]) ), axis = 2) );
for lag in np.arange(numLags):
    SLSCImg[:,:,lag] = SLSC(txFocData, lag+1);
    print('SLSC Lag = '+str(lag+1));
nlagsSLSC = 30;
plt.subplot(1,2,2); imagesc(1000*x_img, 1000*z_img, \
    np.mean(SLSCImg[:,:,:nlagsSLSC], axis=2), (0,1));
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('SLSC'); plt.show()

# VCZ Curves at Two Depths
VCZ1 = np.squeeze(np.mean(np.mean(SLSCImg[75:125,50:100,:], axis=0), axis=0));
VCZ2 = np.squeeze(np.mean(np.mean(SLSCImg[275:325,50:100,:], axis=0), axis=0));
plt.figure(); plt.plot(VCZ1, label='10 mm depth');
plt.plot(VCZ2, label='20 mm depth');
plt.xlabel('lag'); plt.ylabel('Coherence');
plt.legend(); plt.show();
