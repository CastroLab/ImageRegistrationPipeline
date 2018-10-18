function [res, reg, reg_im] = aba_pairwise_reg(im,ref_im, mirt)
%aba_groupwise_reg_pair Returns the groupwise registered mean image

filt = @(im) imgaussfilt(im, 1);
preprocess = @(im) standardizeImage(...
    filt(...
    aba_mask_tissue(im)));
% mean_image = @(im1, im2) mean(cat(3, im1, im2), 3);
%compute_mean_image = @(im_stack) mean(im_stack, 3);

% Output image size
im_size = [480 480];

if isnumeric(im)
    im = preprocess(im);
else
    im = preprocess(imread(im));
end
if size(im, 3) ~= 1
    im = rgb2gray(im);
end

if isnumeric(ref_im)
    ref_im = preprocess(ref_im);
else
    ref_im = preprocess(imread(ref_im));
end

if size(ref_im, 3) ~= 1
    ref_im = rgb2gray(ref_im);
end

im = imresize(im, im_size);
ref_im = imresize(ref_im, im_size);

% Prepare registration array of structs
reg = repmat(...
    struct(...
    'ISH',           [],...
    'fixed',         [],...
    'moving',        [],...
    'rigid_aligned', [],...
    'ref',           [],...
    'mirt_reg',      [],...
    'res',           [],...
    'pre_sim',       [],...
    'mid_sim',       [],...
    'post_sim',      [],...
    'ISH_reg',       []),...
    1, 1);



% Rigid align
[reg.rigid_aligned, ~] = ...
    aba_rigid_align_new(im, ref_im);

reg.ref = ref_im;

% Remove black edges in background
reg.rigid_aligned(reg.rigid_aligned == 0) = 255;
reg.ref(reg.ref == 0) = 255;

% Crop
reg.rigid_aligned = aba_imcrop_edge(reg.rigid_aligned);
reg.ref = aba_imcrop_edge(reg.ref);

% Resize
reg.rigid_aligned = imresize(reg.rigid_aligned, im_size);
reg.ref = imresize(reg.ref, im_size);

reg.rigid_aligned(reg.rigid_aligned == 0) = nan;
reg.ref(reg.ref == 0) = nan;


%figure

[res, reg_im] = mirt2D_register(ref_im,im, mirt.main, mirt.optim);

%res = mirt2Dgroup_sequence(rig_im_stack, mirt.main, mirt.optim);
%mirt_stack = mirt2Dgroup_transform(rig_im_stack, res);

%output_file = ['templates/groupwise', ...
%    '_okno_',     num2str(mirt.main.okno), ...
%    '_sim_',      mirt.main.similarity, ...
%    '_subdiv_',   num2str(mirt.main.subdivide), ...
%    '_lambda_',   num2str(mirt.main.lambda), ...
%    '_maxstep_',  num2str(mirt.optim.maxsteps), ...
%    '_gamma_',    num2str(mirt.optim.gamma), ...
%    '_maxcycle_', num2str(mirt.optim.maxcycle), ...
%    '.mat'];
%save(output_file, 'mirt_stack', 'rig_im_stack', 'im_stack','mean_image', 'res');

end

