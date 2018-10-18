%% Groupwise registration of ABA images using MIRT classification in the ABA dataset
%
% Author: Alex Andonian
% Date: Dec 4, 2016
%
% -------------------------------------------------------------------------
%% Configuration
% -------------------------------------------------------------------------

% Data
conf.imDir ='/Users/aandonia/aba/classification/aba_template_images/';
conf.dataDir = '/Users/aandonia/aba/registration/groupwise/data/';
conf.numTrain = 15;
conf.numTest = 15;
conf.numClasses = 2;
conf.numWords = 1000;
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
conf.prefix = 'template/1000_words';
split = strsplit(conf.prefix, '/');
dir_name = fullfile(conf.dataDir, split{1:end-1});
if ~exist(dir_name, 'dir')
    mkdir(dir_name);
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
conf.distPath = fullfile(conf.dataDir, [conf.prefix '-dists.mat']);
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']);
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']);

randn('state',conf.randSeed);
rand('state',conf.randSeed);
vl_twister('state',conf.randSeed);



% ------------------------------------------------------------------------
%% Setup data
% -------------------------------------------------------------------------

classes = dir(conf.imDir);
classes = classes([classes.isdir]);
classes = {classes(3:conf.numClasses+2).name};

images = {};
imageClass = {};
for ci = 1:length(classes)
    ims = dir(fullfile(conf.imDir, classes{ci}, '*.jpg'))';
    ims = vl_colsubset(ims, conf.numTrain + conf.numTest);
    ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false);
    images = {images{:}, ims{:}};
    imageClass{end+1} = ci * ones(1,length(ims));
end
selTrain = find(mod(0:length(images)-1, conf.numTrain+conf.numTest) < conf.numTrain);
selTest = setdiff(1:length(images), selTrain);
imageClass = cat(2, imageClass{:});
conf.numImages = length(images);

% -------------------------------------------------------------------------
%% Setup model
% -------------------------------------------------------------------------

model.classes = classes;
model.phowOpts = conf.phowOpts;
model.numSpatialX = conf.numSpatialX;
model.numSpatialY = conf.numSpatialY;
model.quantizer = conf.quantizer;
model.vocab = [];
model.w = [];
model.b = [];
model.classify = @classify;

% Prepare registration array of structs
reg = repmat(...
    struct(...
    'ISH',           [],...
    'fixed',         [],...
    'moving',        [],...
    'rigid_aligned', [],...
    'tform',         [],...
    'mirt_reg',      [],...
    'res',           [],...
    'pre_sim',       [],...
    'mid_sim',       [],...
    'post_sim',      [],...
    'ISH_reg',       []),...
    conf.numImages, 1);

% MIRT parameters
main.okno=20;            % mesh window size
main.similarity='ms';    % similarity measure 
main.subdivide = 3;      % number of hierarchical levels
main.lambda = 0.075;       % regularization weight, 0 for none
main.alpha=0.1;          % similarity measure parameter
main.single=1;           % don't show the mesh at each iteration

% MIRT Optimization parameters
optim.maxsteps = 40;    % maximum number of iterations (for a given frame being registered to the mean image)
optim.fundif = 1e-5;    % tolerance (for a given frame)
% optim.gamma = 10;      % initial optimization step size
optim.gamma = 7.5;      % initial optimization step size
optim.anneal=0.8;       % annealing rate
optim.imfundif=1e-6;    % Tolerance of the mean image (change of the mean image)
optim.maxcycle=30;      % maximum number of cycles (at each cycle all frames are registered to the mean image)

% a(a==0)=nan;             % set zeros to nans, the black color here corresponds to the border
                         % around the actual images. We exclude this border by
                         % setting it to NaNs.

mirt.main = main;
mirt.optim = optim;

% ------------------------------------------------------------------------
%% Train vocabulary
% -------------------------------------------------------------------------

if ~exist(conf.vocabPath) || conf.clobber
    
    % Get some PHOW descriptors to train the dictionary
    selTrainFeats = vl_colsubset(selTrain, 30);
    descrs = {};
    
    %for ii = 1:length(selTrainFeats)
    parfor ii = 1:length(selTrainFeats)
        im = imread(fullfile(conf.imDir, images{selTrainFeats(ii)}));
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
    parfor ii = 1:length(images)
        % for ii = 1:length(images)
        fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images));
        im = imread(fullfile(conf.imDir, images{ii}));
        hists{ii} = getImageDescriptor(model, im);
    end
    
    hists = cat(2, hists{:});
    save(conf.histPath, 'hists');
else
    load(conf.histPath);
end

% -------------------------------------------------------------------------
%% Compute feature map
% -------------------------------------------------------------------------

psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5);

% -------------------------------------------------------------------------
%% Train SVM
% -------------------------------------------------------------------------

if ~exist(conf.modelPath) || conf.clobber
    switch conf.svm.solver
        case {'sgd', 'sdca'}
            lambda = 1 / (conf.svm.C *  length(selTrain));
            w = [];
            parfor ci = 1:length(classes)
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
%% Compute pairwise distances between images
% -------------------------------------------------------------------------

if ~exist(conf.distPath, 'file') || conf.clobber
    
    dists.hists_tree = vl_kdtreebuild(hists);
    dists.hists_NxD = hists';
    dists.D = pdist(dists.hists_NxD);
    dists.Y = linkage(dists.D);
    dists.SD = squareform(dists.D);
    [dists.sortedSD, dists.idx] = sort(dists.SD);
    
    save(conf.distPath, 'dists');
else
    load(conf.distPath)
end
% -------------------------------------------------------------------------
%% Nearest neighbors
% -------------------------------------------------------------------------

read = @(i) imread(fullfile(conf.imDir, images{i}));
filt = @(im) imgaussfilt(im, 1);
aba_load = @(i) rgb2gray(...
                standardizeImage(...
                filt(...
                aba_subtract_back(...
                read(i)))));
aba_thresh = @(i) im2double(~im2bw(aba_load(i), graythresh(aba_load(i))));
get_nn_idx = @(i, dists) dists.idx(2, i);
get_nn_dist = @(i, dists) dists.sortedSD(2, i);
load_NN = @(i, dists) aba_load(get_nn_idx(i, dists));
load_thresh_NN = @(i, dists) aba_thresh(get_nn_idx(i, dists));

im_idx = 1;

figure
subplot(1,2,1);
imshow(aba_load(im_idx))
title(['Image: ', num2str(im_idx)]);
subplot(1,2,2);
imshow(load_NN(im_idx, dists));
title(['Image: ', num2str(get_nn_idx(im_idx, dists)), ' with distance:',...
       num2str(get_nn_dist(im_idx, dists))]);

% -------------------------------------------------------------------------
%% Compute mean/median image (grayscale & binary)
% -------------------------------------------------------------------------

mean_image = @(im1, im2) mean(cat(3, im1, im2), 3);
med_image = @(im1, im2) median(cat(3, im2double(im1), im2double(im2)), 3);

im_idx = 1;
im1 = aba_load(im_idx);
im2 = load_NN(im_idx, dists);
im3 = aba_load(dists.idx(3, im_idx));
bim1 = aba_thresh(im_idx);
bim2 = load_thresh_NN(im_idx, dists);

mean_gray = mean_image(im1, im2);
mean_bin = mean_image(bim1, bim2);
med_gray = med_image(im1, im2);
med_bin = med_image(bim1, bim2);

mean_trip = mean(cat(3, im1, im2, im3), 3);

% -------------------------------------------------------------------------
%% Plot nearest neighbors & mean image (grayscale & binary)
% -------------------------------------------------------------------------

figure
subplot(2, 3, 1)
imshow(im1)
title('Im10')
subplot(2, 3, 2)
imshow(mean_gray)
% imshow(med_gray)
title('Mean image')
subplot(2, 3, 3)
imshow(im2)
title('Nearest Neighbor to Im10');
subplot(2, 3, 4)
imshow(bim1)
subplot(2, 3, 5)
imshow(mean_bin)
% imshow(med_bin)
subplot(2, 3, 6)
imshow(bim2)

figure
subplot(1, 4, 1)
imshow(im1)
title('Im1')
subplot(1, 4, 2)
imshow(im2)
title('Im2')
subplot(1, 4, 3)
imshow(im3)
title('Im3')
subplot(1, 4, 4)
imshow(mean_trip)
title('Mean Image')


% -------------------------------------------------------------------------
%% Rigid align to each other and mean image
% -------------------------------------------------------------------------

% Grayscale registration
[rig.im1_to_mean.im, rig.im1_to_mean.tf] = aba_rigid_align_new(im1, mean_trip);
[rig.im2_to_mean.im, rig.im2_to_mean.tf] = aba_rigid_align_new(im2, mean_trip);
[rig.im3_to_mean.im, rig.im3_to_mean.tf] = aba_rigid_align_new(im3, mean_trip);



% -------------------------------------------------------------------------
%% Remove background from grayscale images
% -------------------------------------------------------------------------

back= @(im, tf) ~imwarp(true(size(im)), tf) &~imclearborder(~imwarp(true(size(im)), tf));
last = @(A) A(:, end-1);
Mwarp = ~imwarp(true(size(rig.im1_to_mean.imwb)), rig.im1_to_mean.tf);
% rig_im2_to_im1(Mwarp &~imclearborder(Mwarp)) = 255;

% Remove Background
rig.im1_to_mean.imwb = rig.im1_to_mean.im;
rig.im2_to_mean.imwb = rig.im2_to_mean.im;
rig.im3_to_mean.imwb = rig.im3_to_mean.im;
rig.im1_to_mean.imwb(rig.im1_to_mean.imwb == 0) = 255; 
rig.im2_to_mean.imwb(rig.im2_to_mean.imwb == 0) = 255;
rig.im3_to_mean.imwb(rig.im3_to_mean.imwb == 0) = 255;

rig.im1_to_mean.imwb = aba_imcrop_edge(rig.im1_to_mean.imwb);
rig.im2_to_mean.imwb = aba_imcrop_edge(rig.im2_to_mean.imwb);
rig.im3_to_mean.imwb = aba_imcrop_edge(rig.im3_to_mean.imwb);


% rig.im1_to_mean.imwb(last(back(rig.im1_to_mean.im, rig.im1_to_mean.tf))) = 255;
% rig.im2_to_mean.imwb(last(back(rig.im2_to_mean.im, rig.im2_to_mean.tf))) = 255;
% rig.im1_to_im2.imwb(last(back(rig.im1_to_im2.im, rig.im1_to_im2.tf))) = 255;
% rig.im2_to_im1.imwb(last(back(rig.im2_to_im1.im, rig.im2_to_im1.tf))) = 255;


% -------------------------------------------------------------------------
%% View rigid alignment to each other and mean image
% -------------------------------------------------------------------------

figure
subplot(3, 3, 1)
imshow(im1)
subplot(3, 3, 2)
imshow(mean_gray)
subplot(3, 3, 3)
imshow(im2)
subplot(3, 3, 4)
imshow(rig.im1_to_mean.im)
subplot(3, 3, 5)
imshow(mean_gray)
subplot(3, 3, 6)
imshow(rig.im2_to_mean.im)
subplot(3, 3, 7)
imshow(rig.im1_to_im2.im)
subplot(3, 3, 9)
imshow(rig.im2_to_im1.im)

figure
subplot(3, 3, 1)
imshow(im1)
title('Im1')
subplot(3, 3, 2)
imshow(mean_gray)
title('Mean image')
subplot(3, 3, 3)
imshow(im2)
title('Im2')
subplot(3, 3, 4)
imshow(rig.im1_to_mean.imwb)
title('Im1 rigid aligned to mean image')
subplot(3, 3, 5)
imshow(mean_image(rig.im1_to_mean.im, rig.im2_to_mean.im))
title('Mean image')
subplot(3, 3, 6)
imshow(rig.im2_to_mean.imwb)

subplot(3, 3, 7)
imshow(rig.im1_to_im2.imwb)
subplot(3, 3, 9)
imshow(rig.im2_to_im1.imwb)

figure
subplot(3, 3, 1)
imshow(bim1)
subplot(3, 3, 2)
imshow(mean_bin)
subplot(3, 3, 3)
imshow(bim2)
subplot(3, 3, 4)
imshow(brig.im1_to_mean.im)
subplot(3, 3, 5)
imshow(mean_bin)
subplot(3, 3, 6)
imshow(brig.im2_to_mean.im)
subplot(3, 3, 7)
imshow(brig.im1_to_im2.im)
subplot(3, 3, 9)
imshow(brig.im2_to_im1.im)

% -------------------------------------------------------------------------
%% Groupwise non-linear registration (using MIRT) on rigid aligned to mean
% -------------------------------------------------------------------------

a1 = cat(3, imresize(imadjust(rig.im1_to_mean.imwb), [480 480]),...
            imresize(imadjust(rig.im2_to_mean.imwb), [480 480]),...
            imresize(imadjust(rig.im3_to_mean.imwb), [480 480]));
a1 = im2double(a1);       
a1(a1==255) = nan;

res1=mirt2Dgroup_sequence(a1, main, optim);  % find the transformation (all to the mean)
b1=mirt2Dgroup_transform(im2double(a1), res1);

%%
figure
subplot(3, 3, 1)
imshow(a1(:, :, 1))
title('Im1')
subplot(3, 3, 2)
imshow(a1(:, :, 2))
title('Im2')
subplot(3, 3, 3)
imshow(a1(:, :, 3))
title('Im2')
subplot(3, 3, 4)
imshow(b1(:, :, 1))
title('Im1')
subplot(3, 3, 5)
imshow(b1(:, :, 2))
title('Im2')
subplot(3, 3, 6)
imshow(b1(:, :, 3))
title('Im3')
subplot(3, 3, 7)
imshow(mean(cat(3,  a1(:, :, 1), a1(:, :, 2), a1(:, :, 3)), 3))
title('Pre-reg mean')
subplot(3, 3, 8)
imshow(mean(cat(3,  b1(:, :, 1), b1(:, :, 2), b1(:, :, 3)), 3))
title('Post-reg mean')

figure
title('Im1')
title('Mean Image Pre-Mirt')
subplot(3, 3, 3)
imshow(a1(:, :, 2))
title('Im2')
subplot(3, 3, 4)
imshow(b1(:, :, 1))
title('Im1 Post-Mirt')
subplot(3, 3, 5)
imshow(mean(b1, 3))
title('Mean Image Post-Mirt')
subplot(3, 3, 6)
imshow(b1(:, :, 2))
title('Im2 Post-Mirt')
subplot(3, 3, 8)
imshowpair(b1(:, :, 1), b1(:, :, 2))

% -------------------------------------------------------------------------
%% MIRT on images edge-detected images
% -------------------------------------------------------------------------

a2 = cat(3, imresize(imadjust(im2double(edge(imadjust(rig.im1_to_mean.imwb), 'canny'))), [480 480]),...
            imresize(imadjust(im2double(edge(imadjust(rig.im2_to_mean.imwb), 'canny'))), [480 480]));
a2 = im2double(a2);    
a2(a2==255) = nan;

res2=mirt2Dgroup_sequence(a2, main, optim);  % find the transformation (all to the mean)
b2=mirt2Dgroup_transform(im2double(a2), res2);
b2_g = mirt2Dgroup_transform(imstack, res2);

%%
figure
subplot(3, 3, 1)
imshow(a2(:, :, 1))
title('Im1')
subplot(3, 3, 2)
imshow(mean(a2, 3))
title('Mean Image Pre-Mirt')
subplot(3, 3, 3)
imshow(a2(:, :, 2))
title('Im2')
subplot(3, 3, 4)
imshow(b2(:, :, 1))
title('Im1 Post-Mirt')
subplot(3, 3, 5)
imshow(mean(b2, 3))
title('Mean Image Post-Mirt')
subplot(3, 3, 6)
imshow(b2(:, :, 2))
title('Im2 Post-Mirt')
subplot(3, 3, 7)
imshow(b2_g(:, :, 1));
title('ISH Im1 Post-Mirt')
subplot(3, 3, 8)
imshow(mean_image(b2_g(:, :, 1), b2_g(:, :, 2)))
title('Mean Image')
subplot(3, 3, 9)
imshow(b2_g(:, :, 2))
title('ISH Im2 Post-Mirt')
% -------------------------------------------------------------------------
%% MIRT on images smoothed edge-detected images
% -------------------------------------------------------------------------
sedi1 = imresize(imadjust(im2double(edge(imadjust(rig.im1_to_mean.imwb), 'canny'))), [480 480]);
sedi2 = imresize(imadjust(im2double(edge(imadjust(rig.im2_to_mean.imwb), 'canny'))), [480 480]);
filt = fspecial('average', 25);
sedi1 = imfilter(im2double(sedi1), filt, 'replicate');
sedi2 = imfilter(im2double(sedi2), filt, 'replicate');
sedi1 = imadjust(sedi1);
sedi2 = imadjust(sedi2);

a3 = cat(3, sedi1, sedi2);
a3 = im2double(a3);       
a3(a3==255) = nan;

res3=mirt2Dgroup_sequence(a3, main, optim);  % find the transformation (all to the mean)
b3=mirt2Dgroup_transform(im2double(a3), res3); 

%%

figure
subplot(3, 3, 1)
imshow(a3(:, :, 1))
title('Im1')
subplot(3, 3, 2)
imshow(mean(a3, 3))
title('Mean Image Pre-Mirt')
subplot(3, 3, 3)
imshow(a3(:, :, 2))
title('Im2')
subplot(3, 3, 4)
imshow(b3(:, :, 1))
title('Im1 Post-Mirt')
subplot(3, 3, 5)
imshow(mean(b3, 3))
title('Mean Image Post-Mirt')
subplot(3, 3, 6)
imshow(b3(:, :, 2))
title('Im2 Post-Mirt')
subplot(3, 3, 8)
imshowpair(b3(:, :, 1), b3(:, :, 2))

