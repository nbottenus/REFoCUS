% REFOCUS_DECODE Decode focused beams using the applied delays
%
% rf_decoded = REFOCUS_DECODE(rf_encoded,s_shift)
%
% Parameters:
%   rf_encoded - RF data - samples x receive channel x transmit event
%   delays - Applied delays in samples - transmit event x transmit element
%
% Name/value pairs:
%   'fun' - Inverse function (default = @Hinv_adjoint)
%   'apod' - Apodization applied for each transmit (same size as s_shift)
%   'param' - Parameter for the inverse function
function rf_decoded = refocus_decode(rf_encoded,delays,varargin)

p=inputParser;
p.addOptional('fun',@Hinv_adjoint);
p.addOptional('apod',[]);
p.addOptional('param',[]);
p.parse(varargin{:});

[n_samples, n_receives, n_transmits]=size(rf_encoded);
n_elements=size(delays,2);
assert(size(delays,1)==n_transmits,'Transmit count inconsistent between rf_encoded and delays')

% Default apodization is all ones
if(isempty(p.Results.apod))
    apod = ones(size(delays));
else
    apod = p.Results.apod;
    assert(all(size(apod)==size(delays)),'Apodization size should match delays size')
end

% Promote to floating point if needed
if(~isfloat(rf_encoded))
    rf_encoded=single(rf_encoded);
end

% 1-D FFT to convert time to frequency
RF_encoded=fft(single(rf_encoded));
RF_encoded=permute(RF_encoded,[3 2 1]); % (transmit event x receive channel x time sample)
frequency=(0:n_samples-1)/n_samples;

% Apply encoding matrix at each frequency
RF_decoded = zeros(n_samples,n_elements,n_receives,'like',rf_encoded);
parfor i=2:ceil(n_samples/2) % only compute half, assume symmetry, skip 0 frequency
    Hinv = p.Results.fun(delays,frequency(i),apod,p.Results.param);
    RF_decoded(i,:,:) = Hinv*RF_encoded(:,:,i);
end
RF_decoded=permute(RF_decoded,[1 3 2]); % (frequency x receive channel x transmit element)

% Inverse FFT for real signal
rf_decoded=ifft(RF_decoded,'symmetric');