# Setting up all folders we can import from by adding them to python path
import sys, os, pdb
curr_path = os.getcwd();
sys.path.append(curr_path+'/..');
import hdf5storage

# Importing stuff from all folders in python path
import numpy as np
from focusfun import *
from refocus import *

# TESTING CODE FOR FOCUS_DATA Below
import scipy.io as sio
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# Load Channel Data from Walking-Aperture Tramsmits
data_in = loadmat_hdf5('../../Data/DATA_FocTx_20190323_205714.mat'); # Cyst and Lesions Phantom

# Load All Data from File
time = data_in['time'][0];
rxAptPos = data_in['rxAptPos'];
tx_delays = data_in['tx_delays'];
apod = data_in['apod'];
rxdata_multiTx = data_in['rxdata_multiTx'];

# Load Focused Transmit Channel Data
rxdata_multiTx = rxdata_multiTx / np.max(rxdata_multiTx);
nt, nRx, nTx = rxdata_multiTx.shape;
fs = 1/np.mean(np.diff(time));

# Decode Multistatic Data Using Tikhonov Regularization REFoCUS
rf_decoded = refocus_decode(rxdata_multiTx, tx_delays.T*fs, \
    fun = Hinv_adjoint, apod = apod, param = lambda f: 1);

# Hilbert Transform the Decoded Dataset
full_synth_data = hilbert(rf_decoded, axis = 0);

# Passband Filter Channel Data
N = 10; # Filter Order
fs = 1/np.mean(np.diff(time));
Wn = np.array([3,12])*(1e6)/(fs/2); # Pass Band
b, a = butter(N, Wn, btype = 'bandpass'); # Filter
full_synth_data = filtfilt(b, a, full_synth_data, axis = 0);

# Save Recovered Multistatic Data
data_out = {};
data_out['time'] = time;
data_out['rxAptPos'] = rxAptPos;
data_out['full_synth_data'] = np.real(full_synth_data);
file_out = 'AdjointRecoveredMultistaticData.mat';
hdf5storage.write(data_out, '.', file_out, matlab_compatible=True);
