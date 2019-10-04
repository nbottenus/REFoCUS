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

# Setup Transmit Imaging Case Here
txAptPos = rxAptPos; # Using same array to transmit and recieve
tx_focDepth = 0.030; # Transmit Focus in [m]
theta = 0.0; # Transmit Angle [rad]
tx_dir = np.array([np.sin(theta), 0, np.cos(theta)]); # Each Transmit Beam is Straight Ahead
tx_origin_x = np.arange(-0.00365, 0.0037, 0.00005); # Transmit Origin in [m]
x = rxAptPos[:,0];

# Transmit Apodization
X_XDCR, TX_ORIGIN_X = np.meshgrid(x, tx_origin_x);
rect = lambda x: np.heaviside(x+1/2,1/2)-np.heaviside(x-1/2,1/2);
sigma_rect = 0.008; # [m]
apod = rect( (X_XDCR-TX_ORIGIN_X)/sigma_rect );

# Get Different Rx Data for Each Tx
rxdata_multiTx = np.zeros((time.size, rxAptPos.shape[0], tx_origin_x.size));
for kk in np.arange(tx_origin_x.size):
    rxdata_multiTx[:,:,kk] = focus_fs_to_TxBeam(time, scat, rxAptPos, txAptPos, \
        (tx_origin_x[kk],0.0,0.0), tx_dir, tx_focDepth, apod[kk,:], 0, c);
    print('Completed Transmit '+str(kk)+' at '+str(tx_origin_x[kk])+' m');

# Save Focused Transmit Data to File
del(scat); data_out = {};
data_out['c'] = c;
data_out['time'] = time;
data_out['rxAptPos'] = rxAptPos;
data_out['txAptPos'] = txAptPos;
data_out['tx_focDepth'] = tx_focDepth;
data_out['theta'] = theta;
data_out['tx_origin_x'] = tx_origin_x;
data_out['apod'] = apod;
data_out['rxdata_multiTx'] = rxdata_multiTx;
file_out = 'phantomFocTxDataWalkingAperture.mat';
sio.savemat(file_out, data_out);
