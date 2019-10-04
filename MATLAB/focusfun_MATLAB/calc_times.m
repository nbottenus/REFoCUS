function foc_times = calc_times(foci,elempos,varargin)

% calc_times - computes focusing times
%
% The function computes the (Tx or Rx) time of arrival for specified focal points
% given the array element positions.

% NOTE: Primarely intended when Tx and Rx apertures are the same (i.e. no full synthetic aperture)
%
% INPUTS:
% foci              - M x 3 matrix with position of focal points of interest [m]
% elempos           - N x 3 matrix with element positions [m]
% dc                - time offset [s]; scalar, N x 1 vector, or M x N array
% speed_of_sound    - speed of sounds [m/s]; default 1540 m/s
%
% OUTPUT:
% foc_times         - M x N matrix with times of flight for all foci and all array elements
%

if nargin == 2
    dc = 0;
    speed_of_sound = 1540;
elseif nargin == 3
    dc = varargin{1};
    speed_of_sound = 1540;
elseif nargin == 4
    dc = varargin{1};
    speed_of_sound = varargin{2};
else
    error('Improper argument list');
end

% check for the number of non-singelton dims (i.e. if it's a vector)
% do not change if scalar or matrix

if sum(size(dc)~=1) == 1
    dc = dc(:);
    dc = repmat(dc', [size(foci,1), 1]);
end

foci_tmp = repmat(reshape(foci,size(foci,1),1,3),[1,size(elempos,1),1]);
elempos_tmp = repmat(reshape(elempos,1,size(elempos,1),3),[size(foci_tmp,1),1,1]);

r = foci_tmp - elempos_tmp;

distance = sqrt(sum(r.^2,3));
%distance = sqrt(sum(r.^2,3)) - sqrt(sum(foci_tmp.^2,3));

foc_times = distance./speed_of_sound + dc;
%foc_times = distance./speed_of_sound + sqrt(sum(foci_tmp.^2,3))./speed_of_sound*2 + dc;

return;


