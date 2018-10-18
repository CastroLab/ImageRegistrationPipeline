%% Apply t-sne manifold embedding to the fc7 output of ABA images
% This script performs t-sne embedding on the outputs of a model.
% For CNNs, t-sne embedding is performed on the fc7 layer outputs.
% For SVMs, t-sne embedding is performed on the SIFT/PHOW features.
% The embeddings are saved in ../data.

% ------------------------------------------------------------------------------

run(fullfile(vl_rootnn, 'matlab', 'vl_setupnn'))

PERPLEXITY = 20;
DATADIR = fullfile('D:\', 'METHODS_PAPER', 'feature_extraction', 'data');
SIFTFeatures = fullfile(DATADIR, 'featSIFT.mat');
load(SIFTFeatures)

%%
embedding = tsne(hists', [], 2, 50, PERPLEXITY);

save(fullfile(get_root(), 'embedding', 'data', 'SIFTembedding.mat'))


