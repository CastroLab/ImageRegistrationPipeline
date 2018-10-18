function [out] = aba_detect_points(I, varargin)
%aba_detect_points Detects POI based on desired feature extractor
%   Detailed explanation goes here

% Parse input
p = inputParser;
defaultExtractor = 'PHOW';
validExtractor = {'SIFT', 'SURF', 'PHOW', 'HARRIS'};
checkExtractor = @(x) any(validatestring(x, validExtractor));
addRequired(p, 'I', @isnumeric);
addOptional(p, 'extractor', defaultExtractor, checkExtractor);
parse(p, I, varargin{:});



% Ensure a grayscale image
if size(I, 3) == 3
    I = rgb2gray(I);
end

switch lower(p.Results.extractor)
    
    case 'phow'
        disp('Using PHOW features.')
        
        out = struct('f', [], 'd', []);
        [out.f, out.d] = vl_phow(I);
        
    case 'surf'
        disp('Using SURF features.')
        
    case 'harris'
        disp('Using HARRIS features.')
        
        % Extractor not supported
    otherwise
        error(['Similarity measure is not valid. '...
            'Supported values are: ', strjoin(validExtractor, ', ')])
end
end

