%% Apply t-sne manifold embedding to the fc7 output of ABA images
% This script performs t-sne embedding on the outputs of a model.
% For CNNs, t-sne embedding is performed on the fc7 layer outputs.
% For SVMs, t-sne embedding is performed on the SIFT/PHOW features.
% The embeddings are saved in ../data.

% ------------------------------------------------------------------------------

run(fullfile(vl_rootnn, 'matlab', 'vl_setupnn'))

PERPLEXITY = 20;
DATADIR = fullfile('D:\', 'METHODS_PAPER', 'feature_extraction', 'data');
CNNFeatures = fullfile(DATADIR, 'featCNN_pp_color.mat');
load(CNNFeatures)

%% Load TSNE Embedding and Plot


    DATADIR = fullfile('D:\', 'METHODS_PAPER', 'embedding', 'data');
    FILENAME = fullfile('CNNembedding_pp_color.mat');
    load(fullfile(DATADIR, FILENAME))

    figure, scatter(embedding(:,1), embedding(:,2), '.')


X = embedding;
good_idx = find(embedding(:,1)>=40 & embedding(:,2)<=0);
good_fc7Features = fc7Features(good_idx,:);

load('D:\Preprocess\new_ppImageMaster.mat')

ImageMaster_pp_color_closed_mcl_subset = ppImageMaster(good_idx,:);
%%
embedding = tsne(good_fc7Features, [], 2, 50, PERPLEXITY);

save(fullfile(get_root(), 'embedding', 'data', 'CNNembedding_pp_color_closed_mcl_subset.mat'))


 