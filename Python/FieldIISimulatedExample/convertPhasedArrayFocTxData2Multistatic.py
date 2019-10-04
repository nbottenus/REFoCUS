# Setting up all folders we can import from by adding them to python path
import sys, os, pdb
curr_path = os.getcwd();
sys.path.append(curr_path+'/..');

# Importing stuff from all folders in python path
import numpy as np
from focusfun import *
from refocus import *

# TESTING CODE FOR FOCUS_DATA Below
import scipy.io as sio
from scipy.signal import hilbert
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# Load Channel Data from Phased-Array Transmits
data_in = sio.loadmat('phantomFocTxDataPhasedArray.mat'); # Cyst and Lesions Phantom

# Load All Data from File
c = data_in['c'][0][0];
time = data_in['time'][0];
rxAptPos = data_in['rxAptPos'];
txAptPos = data_in['txAptPos'];
tx_focDepth = data_in['tx_focDepth'][0][0];
theta = data_in['theta'][0];
tx_origin = data_in['tx_origin'][0];
rxdata_multiTx = data_in['rxdata_multiTx'];

# Calculate Relevant Parameters
fs = 1/np.mean(np.diff(time));
x = txAptPos[:,0];
no_elements = x.size;
pitch = np.mean(np.diff(x));

# Method of Recovery
#method = 'Adjoint';
#method = 'RampFilter';
method = 'Tikhonov';

# Load Focused Transmit Channel Data
rxdata_multiTx = rxdata_multiTx / np.max(rxdata_multiTx);
nt, nRx, nTx = rxdata_multiTx.shape;

# Assemble Focal Delays For Each Transmit Beam
tx_delays = np.zeros((txAptPos.shape[0], theta.size));
speed_of_sound = c;
for kk in np.arange(theta.size):
    # Calculate All Geometric Distances
    tx_dir = np.array([np.sin(theta[kk]), 0, np.cos(theta[kk])]);
    tx_center = tx_origin;
    txAptPosRelToCtr = txAptPos - np.tile(tx_center, (txAptPos.shape[0],1));
    txFocRelToCtr = tx_focDepth * np.tile(tx_dir/np.linalg.norm(tx_dir), (txAptPos.shape[0],1));
    txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;
    # Positive Value is Time Delay, Negative is Time Advance
    if np.isinf(tx_focDepth):
        tx_delay = -np.dot(txAptPosRelToCtr,tx_dir)/(speed_of_sound*np.linalg.norm(tx_dir));
    else:
        tx_delay = (np.sqrt(np.sum(txFocRelToCtr**2, axis=1)) - \
            np.sqrt(np.sum(txFocRelToAptPos**2, axis=1)))/speed_of_sound; # Column Vector
    tx_delays[:, kk] = tx_delay;

# Decode Multistatic Data Using REFoCUS
if method == 'Adjoint':
    rf_decoded = refocus_decode(rxdata_multiTx, tx_delays.T*fs, \
        fun = Hinv_adjoint, param = lambda f: 1);
elif method == 'RampFilter':
    rf_decoded = refocus_decode(rxdata_multiTx, tx_delays.T*fs, \
        fun = Hinv_adjoint, param = lambda f: pitch*np.mean(np.diff(theta))*f/c);
elif method == 'Tikhonov':
    rf_decoded = refocus_decode(rxdata_multiTx, tx_delays.T*fs, \
        fun = Hinv_tikhonov, param = 1e-1);

# Hilbert Transform the Decoded Dataset
full_synth_data = hilbert(rf_decoded, axis = 0);

## Compare Recovered and Ground-Truth Multistatic Channel Data

# Ground Truth Multistatic Channel Data
scat = loadmat_hdf5('../../Data/multistaticDataFieldII.mat')['scat'];

# Show Time-Gain-Compensated Channel Data
show_tx_channel = 1;
tgc = np.meshgrid(np.arange(no_elements), time, np.arange(no_elements))[1];
gnd_truth = scat * tgc;
gnd_truth = gnd_truth/np.max(gnd_truth);
recon = np.real(full_synth_data) * tgc;
std_dev = np.std(gnd_truth);

# Normalize for Plotting
recon = recon * np.dot(gnd_truth.flatten(), recon.flatten()) / np.dot(recon.flatten(), recon.flatten());

# Compare Recovered Channel Data to Ground Truth
plt.figure();
plt.subplot(1,2,1);
imagesc(x, time, np.squeeze(gnd_truth[:,:,show_tx_channel]), (-std_dev, std_dev), aspect='auto');
plt.xlabel('Lateral [m]'); plt.ylabel('Time (s)'); plt.title('True Channel Data');
plt.subplot(1,2,2);
imagesc(x, time, np.squeeze(recon[:,:,show_tx_channel]), (-std_dev, std_dev), aspect='auto');
plt.xlabel('Lateral [m]'); plt.ylabel('Time (s)'); plt.title('Reconstructed Channel Data');
plt.show();

# Show Single Channel
show_rx_channel = 48;
plt.figure();
plt.plot(time, np.squeeze(gnd_truth[:,show_rx_channel,show_tx_channel]), 'r', label = 'True Channel Data');
plt.plot(time, np.squeeze(recon[:,show_rx_channel,show_tx_channel]), 'b', label = 'Reconstructed Channel Data');
plt.xlabel('Time (s)'); plt.title('Single Channel Traces'); plt.legend(); plt.show();

# Time-Domain Reconstruction Error
cov_recon_true = np.cov(recon.flatten(), gnd_truth.flatten());
corr_recon_gnd_truth = np.sqrt((cov_recon_true[0,1]*cov_recon_true[1,0]) / \
    (cov_recon_true[0,0]*cov_recon_true[1,1]));
print('Correlation of Recovered Multistatic Dataset with Ground Truth is: '+str(corr_recon_gnd_truth));

## Reconstruct Multistatic Synthetic Aperture Image

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
txFocData = focus_fs(time, full_synth_data, foc_pts, rxAptPos, rxAptPos, 0, 0, c);
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
plt.figure();
plt.plot(VCZ1, label='10 mm depth');
plt.plot(VCZ2, label='20 mm depth');
plt.xlabel('lag'); plt.ylabel('Coherence');
plt.legend(); plt.show();
