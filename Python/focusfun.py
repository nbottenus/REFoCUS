import numpy as np
import pdb

def calc_times(foci, elempos, dc = 0, speed_of_sound = 1540):
    ''' foc_times = calc_times(foci, elempos, dc = 0, speed_of_sound = 1540)

    CALC_TIMES - computes focusing times

    The function computes the (Tx or Rx) time of arrival for specified focal points
    given the array element positions.

    NOTE: Primarily intended when Tx and Rx apertures are the same (i.e. no full synthetic aperture)

    INPUTS:
    foci              - M x 3 matrix with position of focal points of interest [m]
    elempos           - N x 3 matrix with element positions [m]
    dc                - time offset [s]; scalar, N x 1 vector, or M x N array
    speed_of_sound    - speed of sounds [m/s]; default 1540 m/s

    OUTPUT:
    foc_times         - M x N matrix with times of flight for all foci and all array elements '''

    if type(dc).__module__ == 'builtins':
        dc = np.array([dc]);
    if not(np.isscalar(dc)) and sum(np.array(dc.shape)==1) <= 1:
        np.tile(dc, (foci.shape[0], 1));

    foci_tmp = np.tile(np.reshape(foci,(foci.shape[0],1,3)), (1,elempos.shape[0],1));
    elempos_tmp = np.tile(np.reshape(elempos,(1,elempos.shape[0],3)), (foci_tmp.shape[0],1,1));

    r = foci_tmp - elempos_tmp;

    distance = np.sqrt(np.sum(r**2, axis = 2));
    foc_times = distance/speed_of_sound + dc;

    return foc_times;

def focus_fs_to_TxBeam(t, signal, rxAptPos, txAptPos, tx_center, tx_dir, tx_focDepth, tx_apod, dc_tx = 0, speed_of_sound = 1540):
    ''' foc_data = focus_fs_to_TxBeam(t, signal, rxAptPos, txAptPos, tx_center, tx_dir, tx_focDepth, tx_apod, dc_tx = 0, speed_of_sound = 1540)

    FOCUS_FS_TO_TXBEAM - Focuses the RF data at desired locations

    The function interpolates the RF signals collected using the full synthetic sequence
    to focus the data at desired locations

    INPUTS:
    t                  - T x 1 time vector for samples of the input signal
    signal             - T x N x M matrix containing input RF data to be interpolated
    rxAptPos           - N x 3 matrix with positions of the Rx apertures (elements) [m]
    txAptPos           - M x 3 matrix with positions of the Tx apertures (elements) [m]
    tx_center          - 1 x 3 vector with the position of the center of the Tx aperture [m]
    tx_dir             - 1 x 3 matrix with direction of transmit beam
    tx_focDepth        - Depth of transmit focus along transmit direction [m]
    tx_apod            - M x 1 vector of apodizations for transmit beam
    dc_tx              - time offsets [s] for Tx; scalars, M x 1 vectors
    speed_of_sound     - speed of sounds [m/s]; default 1540 m/s

    OUTPUT:
    foc_data           - T x N vector with transmit-beamformed data points '''

    if np.isscalar(dc_tx): dc_tx = np.array(dc_tx); # make dc_tx have array type

    # calculate all these relative distances to do retrospective transmit focusing
    txAptPosRelToCtr = txAptPos - np.tile(tx_center, (txAptPos.shape[0], 1));
    txFocRelToCtr = tx_focDepth * np.tile(tx_dir/np.linalg.norm(tx_dir), (txAptPos.shape[0], 1));
    txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;

    # positive value is time delay, negative is time advance
    if np.isinf(tx_focDepth): # Plane Wave Option
        tx_delay = (np.mat(-txAptPosRelToCtr)*np.mat(tx_dir/np.linalg.norm(tx_dir)).T)/speed_of_sound;
    else: # Column Vector
        tx_delay = (np.sqrt(np.sum(txFocRelToCtr**2, axis = 1))-np.sqrt(np.sum(txFocRelToAptPos**2, axis = 1)))/speed_of_sound;
    tx_delay = tx_delay + dc_tx;

    # transmit beamforming on full-synthetic aperture dataset: delayed-and-summed
    foc_data = np.zeros((t.shape[0], rxAptPos.shape[0])).astype('complex128');
    for i in np.arange(rxAptPos.shape[0]):
        for j in np.arange(txAptPos.shape[0]):
            foc_data[:,i] += tx_apod[j] * np.interp(t-tx_delay[j], t, signal[:,i,j], left=0, right=0);

    return foc_data;

def focus_fs(t, signal, foc_pts, rxAptPos, txAptPos = None, dc_rx = 0, dc_tx = 0, speed_of_sound = 1540):
    ''' foc_data = focus_data_fs(t, signal, foc_pts, rxAptPos, txAptPos = rxAptPos, dc_rx = 0, dc_tx = 0, speed_of_sound = 1540)

    FOCUS_DATA_FS - Focuses the RF data at desired locations

    The function interpolates the RF signals collected using the full synthetic sequence
    to focus the data at desired locations.

    INPUTS:
    t                  - T x 1 time vector for samples of the input signal
    signal             - T x N x M matrix containing input RF data to be interpolated
    foc_pts            - P x 3 matrix with position of focal points [m]
    rxAptPos           - N x 3 matrix with positions of the Rx apertures (elements) [m]
    txAptPos           - M x 3 matrix with positions of the Tx apertures (elements) [m]
                       - txAptPos = rxAptPos by default
    dc_rx, dc_tx       - time offsets [s] for Tx and Rx; scalars, N (M) x 1 vectors, or P x N (M) matrix
    speed_of_sound     - speed of sounds [m/s]; default 1540 m/s

    OUTPUT:
    foc_data - P x N x M vector with interpolated (RF) data points '''

    # Set txAptPos to rxAptPos if not set
    if txAptPos is None: txAptPos = rxAptPos;

    # time from the focus to receive  apertures (array elements)
    rx_times = calc_times(foc_pts, rxAptPos, dc = dc_rx, speed_of_sound = speed_of_sound);

    # time from the transmit apertures (array elements) to focus
    tx_times = calc_times(foc_pts, txAptPos, dc = dc_tx, speed_of_sound = speed_of_sound);

    # focused but not summed rf data
    foc_data = np.zeros((foc_pts.shape[0], rxAptPos.shape[0])).astype('complex128');
    for i in np.arange(rx_times.shape[1]):
        for j in np.arange(tx_times.shape[1]):
            foc_data[:,i] += np.interp(rx_times[:,i]+tx_times[:,j], t, signal[:,i,j], left=0, right=0);

    return foc_data;

# Define Loadmat Function for HDF5 Format ('-v7.3 in MATLAB')
import h5py
def loadmat_hdf5(filename):
    file = h5py.File(filename,'r');
    out_dict = {}
    for key in file.keys():
        out_dict[key] = np.ndarray.transpose(np.array(file[key]));
    file.close();
    return out_dict;

# Python-Equivalent Command for IMAGESC in MATLAB
import matplotlib.pyplot as plt
def imagesc(x, y, img, rng, cmap='gray', numticks=(3, 3), aspect='equal'):
    exts = (np.min(x)-np.mean(np.diff(x)), np.max(x)+np.mean(np.diff(x)), \
        np.min(y)-np.mean(np.diff(y)), np.max(y)+np.mean(np.diff(y)));
    plt.imshow(np.flipud(img), cmap=cmap, extent=exts, vmin=rng[0], vmax=rng[1], aspect=aspect);
    plt.xticks(np.linspace(np.min(x), np.max(x), numticks[0]));
    plt.yticks(np.linspace(np.min(y), np.max(y), numticks[1]));
    plt.gca().invert_yaxis(); plt.colorbar();
