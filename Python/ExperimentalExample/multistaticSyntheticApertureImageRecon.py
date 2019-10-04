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
data_in = loadmat_hdf5('AdjointRecoveredMultistaticData.mat'); # Recovered by Adjoint
#data_in = loadmat_hdf5('TikhonovRecoveredMultistaticData.mat'); # Recovered by Tikhonov
time = data_in['time'][0];
scat = data_in['full_synth_data'];
scat_h = hilbert(scat, axis = 0); del(scat);
rxAptPos = data_in['rxAptPos'];
pitch = np.mean(np.diff(rxAptPos[:,0]));
no_elements = rxAptPos.shape[0];

# Take Hilbert Transform of Field II Simulated Data
nt, nRx, nTx = scat_h.shape;

# Points to Focus and Get Image At
dBrange = np.array([-60, 0]);
num_x = 150; num_z = 150;
xlims = np.array([-7e-3, 7e-3]);
x_img = np.linspace(xlims[0], xlims[1], num_x);
zlims = np.array([16e-3, 29e-3]);
z_img = np.linspace(zlims[0], zlims[1], num_z);
c = 1460; # Sound Speed [m/s] in Phantom

# Full Synthetic Aperture Image Reconstruction
(Z, Y, X) = np.meshgrid(z_img, 0, x_img);
foc_pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()));
txFocData = focus_fs(time, scat_h, foc_pts, rxAptPos, rxAptPos, 0, 0, c);
txFocData = txFocData.reshape(z_img.size, x_img.size, rxAptPos.shape[0]);
bModeImg = np.squeeze(np.sum(txFocData, axis = 2));
plt.figure(); imagesc(1000*x_img, 1000*z_img, \
    20*np.log10(np.abs(bModeImg)/np.max(np.abs(bModeImg[:]))), dBrange);
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('DAS Beamforming'); plt.show();
