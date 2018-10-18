%% Compute output features for all models

% Data paths
DATADIR = fullfile('D:\', 'METHODS_PAPER', 'feature_extraction', 'data')
CNNFeatures = fullfile(DATADIR, 'CNNFeatures.mat')
SIFTFeatures = fullfile(DATADIR, 'SIFTFeatures.mat')

% Recompute features?
recompute = false

if ~exist(CNNFeatures) || recompute
    computeCNNFeatures
else
    disp('CNN features have already been computed!')
    disp('Loading CNN features')
    load(CNNFeatures)
end

if ~exist(SIFTFeatures) || recompute
    computeSIFTFeatures
else
    disp('SIFT features have already been computed!')
    disp('Loading SIFT features')
    load(SIFTFeatures)
end