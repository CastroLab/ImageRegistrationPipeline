%% Load test data
aba_root = aba_get_root_dir();
conf.imDir = fullfile(aba_root, 'classification/aba_template_images/');
load(fullfile(aba_root, 'registration/groupwise/data/template/1000_words-model.mat'))
load(fullfile(aba_root, 'registration/groupwise/data/template/1000_words-dists.mat'))

images = model.images;
images = cellfun(@(name) strcat(conf.imDir, name), images, 'UniformOutput', false);

im_idx = 1;
num_nn = 10;
get_n_NearestNeighbors_idx = @(i, n, dists) dists.idx(2:(n+1), i);
nn_idx = get_n_NearestNeighbors_idx(im_idx, num_nn, dists);
nn = {images(nn_idx)};

reg_images = [images{im_idx}, nn{:}];

%%
% MIRT parameters
main.okno       = 25;   % mesh window size
main.similarity = 'ms'; % similarity measure 
main.subdivide  = 4;    % number of hierarchical levels
main.lambda     = 0.5; % regularization weight, 0 for none
main.alpha      = 0.1;  % similarity measure parameter
main.single     = 1;    % don't show the mesh at each iteration

% MIRT Optimization parameters
optim.maxsteps = 50;    % maximum number of iterations (for a given frame being registered to the mean image)
optim.fundif   = 1e-7;  % tolerance (for a given frame)
optim.gamma    = 7.5;   % initial optimization step size
optim.anneal   = 0.8;   % annealing rate
optim.imfundif = 1e-7;  % Tolerance of the mean image (change of the mean image)
optim.maxcycle = 50;    % maximum number of cycles (at each cycle all frames are registered to the mean image)

mirt.main = main;
mirt.optim = optim;


%% Register

reg_im = aba_groupwise_reg(reg_images, mirt);
