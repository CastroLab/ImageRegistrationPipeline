%% Groupwise Registration of 100 nearest neighbor images using CNN distance metric

% ------------------------------------------------------------------------------
% Setup options
% ------------------------------------------------------------------------------

warning off
run(fullfile(vl_rootnn, 'matlab', 'vl_setupnn'))
opts.modelType = 'alexnet';
opts.expDir = fullfile(aba_get_etna_root(), 'mitral_classification', 'exp');
opts.expDir = fullfile(opts.expDir, sprintf('aba-%s', opts.modelType));
net = load(fullfile(opts.expDir, 'net-deployed.mat'));
norm = @(im) imresize(single(im), ...
    net.meta.normalization.imageSize(1:2)) - net.meta.normalization.averageImage;

seed_im = norm(imread('seed_im.jpg'));

% ------------------------------------------------------------------------------
%% Make forward pass with seed image
% ------------------------------------------------------------------------------

im_res = vl_simplenn(net, seed_im, 1);
im_out = im_res(end-2).x;

dist_net = net;
dist_net.layers = dist_net.layers(1:end-2);
dist_net = addCustomLossLayer(dist_net, @l2LossForward, @l2LossBackward);
dist_net.layers{end}.class = im_out;


dataDir = fullfile(aba_get_etna_root, 'mitral_classification', 'images', 'raw');
filePattern = fullfile(dataDir, '*.jpg');
images = dir(filePattern);
num_images = length(images);

im_names = cellfun(@fullfile, {images.folder}, {images.name}, 'UniformOutput', false);

if ~exist('im_stack.mat', 'file')
    im_stack = vl_imreadjpeg(im_names, 'NumThreads', 8, 'Resize', [227 227], ...
        'Pack', 'SubtractAverage', single(net.meta.normalization.averageImage));
    im_stack = im_stack{:};
    save('im_stack.mat', 'im_stack')
else
    load('im_stack.mat')
end

% im_stack = gpuArray(im_stack);
%% Run the CNN to calculate pair-wise distances
f_name = fullfile(aba_get_root_dir, 'registration', 'groupwise', 'tests', 'res.mat');
if ~exist(f_name, 'file')
    res = vl_simplenn(dist_net, im_stack);
    save('res.mat', 'res')              
else
    load('res.mat')
end

%%

dists = squeeze(res(end).x);
[~, nn_idx] = sort(dists);

num_show = 100;
nrows = ceil(sqrt(num_show));

reg_images = {im_names{nn_idx(1:num_show)}};
figure
for i = 1:num_show
    subplot(nrows, nrows, i)
   imshow(imread(im_names{nn_idx(i)})) 
end

%%

mirt = aba_load_mirt_default();

reg_im = aba_groupwise_reg(reg_images, mirt);