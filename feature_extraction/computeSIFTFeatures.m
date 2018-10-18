%% Image classification in the ABA dataset
%   This program demonstrates how to use VLFeat to construct an image
%   classifier on the ABA data. The classifier uses PHOW
%   features (dense SIFT), spatial histograms of visual words, and a
%   Chi2 SVM. To speedup computation it uses VLFeat fast dense SIFT,
%   kd-trees, and homogeneous kernel map. The program also
%   demonstrates VLFeat PEGASOS SVM solver, although for this small
%   dataset other solvers such as LIBLINEAR can be more efficient.
%
%   Call aba_phow to train and test a classifier on a small
%   subset of the ABA data.
%
%   The ABA data is saved into conf.ABADir, which defaults to
%   'data/images'. Change this path to the desired location, for
%   instance to point to an existing copy of the ABA data.
%
%   The program can also be used to train a model on custom data by
%   pointing conf.ABADir to it. Just create a subdirectory for each
%   class and put the training images there. Make sure to adjust
%   CONF.NUMTRAIN accordingly.
%
%   Intermediate files are stored in the directory CONF.DATADIR. All
%   such files begin with the prefix CONF.PREFIX, which can be changed
%   to test different parameter settings without overriding previous
%   results.
%
%   The program saves the trained model in
%   <CONF.DATADIR>/<CONF.PREFIX>-model.mat. This model can be used to
%   test novel images independently of the ABA data.
%
%     load('data/baseline-model.mat'); # change to the model path
%     label = model.classify(model, im);
%
%
% -------------------------------------------------------------------------
%% Configuration
% -------------------------------------------------------------------------

% Data
load(fullfile('D:\', 'ABA', 'images', 'master', 'ImageMaster.mat'))
conf.ABADir = fullfile('D:\', 'ABA' , 'images', 'master', 'ish');
conf.numImages = length(ImageMaster)
conf.dataDir = 'D:\METHODS_PAPER\feature_extraction\data';
conf.autoDownloadData = false;
conf.numTrain = 0.75 * conf.numImages;
conf.numTest = 0.25 * conf.numImages;
conf.numClasses = 6;
conf.numWords = 4096;
conf.numSpatialX = [2 4];
conf.numSpatialY = [2 4];

% Vocab
conf.phowOpts = {'Step', 3};
conf.quantizer = 'kdtree';

% SVM
conf.svm.C = 10;
conf.svm.solver = 'sdca'; % Options = {'sdca', 'sgd', 'liblinear'}
conf.svm.biasMultiplier = 1;

% Misc.
conf.clobber = false;
conf.tinyProblem = false;
conf.randSeed = 1;

% Output
conf.prefix = '4096_words';
split = strsplit(conf.prefix, '/');

for i = 1:length(split) - 1
    dir_name = fullfile(conf.dataDir, split{1:i});
    if ~exist(dir_name)
        mkdir(dir_name);
    end
end

if conf.tinyProblem
    conf.prefix = 'tiny';
    conf.numClasses = 5;
    conf.numSpatialX = 2;
    conf.numSpatialY = 2;
    conf.numWords = 300;
    conf.phowOpts = {'Verbose', 2, 'Sizes', 7, 'Step', 5};
end

conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']);
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']);
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']);
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']);

randn('state',conf.randSeed);
rand('state',conf.randSeed);
vl_twister('state',conf.randSeed);

% -------------------------------------------------------------------------
%% Setup data
% -------------------------------------------------------------------------

classes = dir(conf.ABADir);
classes = classes([classes.isdir]);
%classes = {classes(3:conf.numClasses+2).name};
classes = {classes(3).name};

images = {};
imageClass = {};
for ci = 1:length(classes)
    ims = dir(fullfile(conf.ABADir, classes{ci}, '*.jpg'))';
    ims = vl_colsubset(ims, conf.numTrain + conf.numTest);
    ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false);
    images = {images{:}, ims{:}};
    imageClass{end+1} = ci * ones(1,length(ims));
end
selTrain = find(mod(0:length(images)-1, conf.numTrain+conf.numTest) < conf.numTrain);
selTest = setdiff(1:length(images), selTrain);
imageClass = cat(2, imageClass{:});

model.classes = classes;
model.phowOpts = conf.phowOpts;
model.numSpatialX = conf.numSpatialX;
model.numSpatialY = conf.numSpatialY;
model.quantizer = conf.quantizer;
model.vocab = [];
model.w = [];
model.b = [];
model.classify = @classify;

% ------------------------------------------------------------------------
%% Train vocabulary
% -------------------------------------------------------------------------

if ~exist(conf.vocabPath) || conf.clobber

    % Get some PHOW descriptors to train the dictionary
    selTrainFeats = vl_colsubset(selTrain, 10);
    descrs = {};

    for ii = 1:length(selTrainFeats)
        im = imread(fullfile(conf.ABADir, images{selTrainFeats(ii)}));
        im = standarizeImage(im);
        [~, descrs{ii}] = vl_phow(im, model.phowOpts{:});
    end

    descrs = vl_colsubset(cat(2, descrs{:}), 10e4);
    descrs = single(descrs);

    % Quantize the descriptors to get the visual words
    vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50);
    save(conf.vocabPath, 'vocab');
else
    load(conf.vocabPath);
end

model.vocab = vocab;

if strcmp(model.quantizer, 'kdtree')
    model.kdtree = vl_kdtreebuild(vocab);
end

% -------------------------------------------------------------------------
%% Compute spatial histograms
% -------------------------------------------------------------------------

if ~exist(conf.histPath) || conf.clobber
    hists = {};
    for ii = 1:length(images)
        fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images));
        im = imread(fullfile(conf.ABADir, images{ii}));
                im = standarizeImage(im);
        hists{ii} = getImageDescriptor(model, im);
    end

    hists = cat(2, hists{:});
    save(conf.histPath, 'hists','-v7.3');
else
    load(conf.histPath);
end

% ------------------------------------------------------------------------
%% Compute feature map
% -------------------------------------------------------------------------
psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5);
% ------------------------------------------------------------------------
%% Train SVM
% -------------------------------------------------------------------------

if ~exist(conf.modelPath) || conf.clobber
    switch conf.svm.solver
        case {'sgd', 'sdca'}
            lambda = 1 / (conf.svm.C *  length(selTrain));
            w = [];
            for ci = 1:length(classes)
                perm = randperm(length(selTrain));
                fprintf('Training model for class %s\n', classes{ci});
                y = 2 * (imageClass(selTrain) == ci) - 1;
                [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), lambda, ...
                    'Solver', conf.svm.solver, ...
                    'MaxNumIterations', 50/lambda, ...
                    'BiasMultiplier', conf.svm.biasMultiplier, ...
                    'Epsilon', 1e-3);
            end

        case 'liblinear'
            svm = train(imageClass(selTrain)', ...
                sparse(double(psix(:,selTrain))),  ...
                sprintf(' -s 3 -B %f -c %f', ...
                conf.svm.biasMultiplier, conf.svm.C), ...
                'col');
            w = svm.w(:,1:end-1)';
            b =  svm.w(:,end)';
    end

    model.b = conf.svm.biasMultiplier * b;
    model.w = w;

    save(conf.modelPath, 'model');
else
    load(conf.modelPath);
end

% -------------------------------------------------------------------------
%% Test SVM and evaluate
% -------------------------------------------------------------------------

% Estimate the class of the test images
scores = model.w' * psix + model.b' * ones(1,size(psix,2));
[drop, imageEstClass] = max(scores, [], 1);

% Compute the confusion matrix
idx = sub2ind([length(classes), length(classes)], ...
    imageClass(selTest), imageEstClass(selTest));
confus = zeros(length(classes));
confus = vl_binsum(confus, ones(size(idx)), idx);

% Plots
figure(1); clf;
subplot(1,2,1);
imagesc(scores(:,[selTrain selTest])); title('Scores');
set(gca, 'ytick', 1:length(classes), 'yticklabel', classes);
subplot(1,2,2);
imagesc(confus);
title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
    100 * mean(diag(confus)/conf.numTest) ));
% print('-depsc2', [conf.resultPath '.ps']);

if ~exist([conf.resultPath '.mat']) || conf.clobber
    save([conf.resultPath '.mat'], 'confus', 'conf');
else
    load([conf.resultPath '.mat']);
end

% -------------------------------------------------------------------------
%% Nearest neighbors
% -------------------------------------------------------------------------

hists_tree = vl_kdtreebuild(hists);
[nn, DIST] = vl_kdtreequery(hists_tree, hists, hists(:,1),'NUMNEIGHBORS', 10);

