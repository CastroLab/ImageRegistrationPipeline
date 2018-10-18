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

filt = @(im) imgaussfilt(im, 1);
preprocess = @(im) standardizeImage(...
    filt(...
    aba_mask_tissue(im)));
% mean_image = @(im1, im2) mean(cat(3, im1, im2), 3);
compute_mean_image = @(im_stack) mean(im_stack, 3);

% Output image size
im_size = [480 480];
num_images = length(reg_images);
im_stack = zeros([im_size, num_images]);

% Load images and compute mean
for i = 1:num_images
    im = preprocess(imread(reg_images{i}));
    if size(im, 3) ~= 1
        im = rgb2gray(im);
    end
    
    im_stack(:, :, i) = im;
end

% -------------------------------------------------------------------------
%% Convert the to required format
% -------------------------------------------------------------------------
I = single(rgb2gray(imread(reg_images{1})));
J = single(rgb2gray(imread(reg_images{2})));
im_file = fullfile(aba_get_root_dir(), 'classification/aba_images/',...
    'range1_sample_435-445/cropped_126_Atp6v0a1_437_1909_440ishfull.jpg');
im_file2 = fullfile(aba_get_root_dir(), 'classification/aba_images/',...
    'range1_sample_435-445/cropped_734_Nr5a1_407_2101_437ishfull.jpg');
I = imread(im_file);
J = imread(im_file2);
% I = single(im_stack(:, :, 1));
% J = single(im_stack(:, :, 2));
I = single(rgb2gray(I));
J = single(rgb2gray(J));

% clf;
% imagesc(I)
% axis equal;
% axis off;
% axis tight;

% -------------------------------------------------------------------------
%% Run PHOW
% -------------------------------------------------------------------------

num_points = 10000;
[f, d] = vl_phow(I);
f = f(:, f(3,:) > 25 );
perm = randperm(size(f,2));
% sel  = perm(1:num_points);
sel = perm(1:end);
scatter(f(1, sel), f(2, sel), 1)


%%

[f1, d1] = vl_phow(J);
f1 = f1(:, f1(3,:) > 5 );
perm1 = randperm(size(f1,2));
% sel1  = perm1(1:num_points);
sel1 = perm1(1:end);
figure
scatter(f1(1,sel1), f1(2, sel1), 1)
%%

matches = vl_ubcmatch(f, f1);

scatter(f(

% X = cat(2,f(1, sel), f(2, sel));
% Y = cat(2,f1(1, sel1), f1(2, sel1));
% len = min(size(f, 1), size(f1, 1));
% X = X(1:len, :);
% Y = Y(1:len, :);

%%

% Init full set of options %%%%%%%%%%
opt.method='nonrigid'; % use nonrigid registration
opt.beta=2;            % the width of Gaussian kernel (smoothness)
opt.lambda=8;          % regularization weight

opt.viz=1;              % show every iteration
opt.outliers=0.7;       % use 0.7 noise weight
opt.fgt=0;              % do not use FGT (default)
opt.normalize=1;        % normalize to unit variance and zero mean before registering (default)
opt.corresp=1;          % compute correspondence vector at the end of registration (not being estimated by default)

opt.max_it=100;         % max number of iterations
opt.tol=1e-10;          % tolerance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Transform, C]=cpd_register(X,Y, opt);

figure,cpd_plot_iter(X, Y); title('Before');
figure,cpd_plot_iter(X, Transform.Y, C);  title('After registering Y to X. And Correspondences');