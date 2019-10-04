import numpy as np
import pdb

def H_model_matrix(delays, f, apod):
    ''' H_MODEL_MATRIX Computes H Matrix Based on Delays, Frequency, and Apodization
    H = ForwardModel(delays, f, apod)
    INPUTS:
        delays = N x M matrix (N transmits, M elements) of delays (in samples)
        f = Normalized Frequency (1/samples) Ranging from 0 to 1
        apod = N x M matrix of apodizations
    OUTPUTS:
        H = N x M model matrix '''

    # Model Matrix with Delay, Frequency, and Apodization
    H = np.matrix(apod*np.exp(-1j*2*np.pi*f*delays));
    return H;

def Hinv_adjoint(delays, f, apod, param = lambda f: f):
    ''' H_INV_ADJOINT Computes Adjoint of H Matrix
    Hinv = Hinv_adjoint(delays,f,apod,param)
    INPUTS:
        delays = N x M matrix (N transmits, M elements) of delays (in samples)
        f = Normalized Frequency (1/samples) Ranging from 0 to 1
        apod = N x M matrix of apodizations
        param = set to zero to remove ramp filter (otherwise, don't set this)
    OUTPUTS:
        Hinv = M x N matrix inverse of H '''

    # Forward Model Matrix
    H = H_model_matrix(delays, f, apod);
    # Adjoint (Conjugate Transpose)
    Hinv = H.H;
    # Default (Use Ramp Filter)
    Hinv = param(f)*Hinv;
    return Hinv;

def Hinv_tikhonov(delays, f, apod, param = 1e-2):
    ''' H_INV_TIKHONOV Computes Tikhonov Regularized Inverse of H Matrix
    Hinv = Hinv_tikhonov(delays,f,apod,param)
    INPUTS:
        delays = N x M matrix (N transmits, M elements) of delays (in samples)
        f = Normalized Frequency (1/samples) Ranging from 0 to 1
        apod = N x M matrix of apodizations
        param = Regularization Parameter
    OUTPUTS:
        Hinv = M x N matrix inverse of H '''

    # Forward Model Matrix
    H = H_model_matrix(delays, f, apod);
    # Regularization Relative to Maximum Singular Value
    smax = np.linalg.norm(H, ord=2);
    reg = np.matrix(param*smax*np.eye(H.shape[1]));
    # Tikhonov Regularized Inverse Matrix
    Hinv = np.linalg.solve(H.H*H + reg.H*reg, H.H);
    return Hinv;

def Hinv_rsvd(delays, f, apod, param = 1e-2):
    ''' H_INV_RSVD Computes Regularized SVD-Based Inverse of H Matrix
    Hinv = Hinv_rsvd(delays,f,apod,param)
    INPUTS:
        delays = N x M matrix (N transmits, M elements) of delays (in samples)
        f = Normalized Frequency (1/samples) Ranging from 0 to 1
        apod = N x M matrix of apodizations
        param = Regularization Parameter
    OUTPUTS:
        Hinv = M x N matrix inverse of H '''

    # Forward Model Matrix
    H = H_model_matrix(delays,f,apod);
    # Compute SVD of H
    U, s, VH = np.linalg.svd(H);
    # Regularized Inversion Singular Value Matrix
    sinv = s/(s**2+(param*s[0])**2); # Regularize small values
    Sinv = np.matrix(np.zeros(H.T.shape));
    Sinv[:sinv.size, :sinv.size] = np.diag(sinv);
    # Regularized Inversion of Model Matrix
    Hinv = VH.H*Sinv*U.H;
    return Hinv;

def Hinv_tsvd(delays, f, apod, param = 1e-2):
    ''' H_INV_TSVD Computes Truncated SVD-Based Inverse of H Matrix
    Hinv = Hinv_tsvd(delays,f,apod,param)
    INPUTS:
        delays = N x M matrix (N transmits, M elements) of delays (in samples)
        f = Normalized Frequency (1/samples) Ranging from 0 to 1
        apod = N x M matrix of apodizations
        param = Truncation Parameter
    OUTPUTS:
        Hinv = M x N matrix inverse of H '''

    # Forward Model Matrix
    H = H_model_matrix(delays,f,apod);
    # Compute SVD of H
    U, s, VH = np.linalg.svd(H);
    # Truncate Singular Value Matrix
    sinv = 1/s; sinv[s<param*s[0]] = 0; # Truncate small singular values
    Sinv = np.matrix(np.zeros(H.T.shape));
    Sinv[:sinv.size, :sinv.size] = np.diag(sinv);
    # Regularized Inversion of Model Matrix
    Hinv = VH.H*Sinv*U.H;
    return Hinv;

def refocus_decode(rf_encoded, delays, fun=Hinv_adjoint, apod=None, param=None):
    ''' REFOCUS_DECODE Decode focused beams using the applied delays

    rf_decoded = REFOCUS_DECODE(rf_encoded,s_shift)

    Parameters:
        rf_encoded - RF data - samples x receive channel x transmit event
        delays - Applied delays in samples - transmit event x transmit element

    Name/value pairs:
        'fun' - Inverse function (default = @Hinv_adjoint)
        'apod' - Apodization applied for each transmit (same size as s_shift)
        'param' - Parameter for the inverse function '''

    # Get input dimensions
    n_samples, n_receives, n_transmits = rf_encoded.shape;
    n_elements = delays.shape[1];
    assert(delays.shape[0] == n_transmits), 'Transmit count inconsistent between rf_encoded and delays';

    # Default apodization is all ones
    if apod is None:
        apod = np.ones(delays.shape);
    else:
        assert(apod.shape==delays.shape), 'Apodization size should match delays size';

    # Promote to floating point if needed
    if(not((rf_encoded.dtype == np.dtype('float32')) or (rf_encoded.dtype == np.dtype('float64')))):
        rf_encoded = rf_encoded.astype('float32');

    # 1-D FFT to convert time to frequency
    RF_encoded = np.fft.rfft(rf_encoded.astype('float32'), axis = 0);
    RF_encoded = np.transpose(RF_encoded, axes=(2,1,0)); # (transmit event x receive channel x time sample)
    frequency = np.arange(np.ceil(n_samples/2))/n_samples;

    # Apply encoding matrix at each frequency
    RF_decoded = np.zeros((int(np.ceil(n_samples/2)), n_elements, n_receives), dtype = np.dtype('complex64'));
    for i in np.arange(1,int(np.ceil(n_samples/2))):
        Hinv = fun(delays, frequency[i], apod, param);
        RF_decoded[i,:,:] = np.array(np.dot(Hinv, np.matrix(RF_encoded[:,:,i])));
    RF_decoded = np.transpose(RF_decoded, axes=(0,2,1)); # (frequency x receive channel x transmit element)

    # Inverse FFT for real signal
    rf_decoded = np.fft.irfft(RF_decoded, n=n_samples, axis = 0);
    return rf_decoded;
