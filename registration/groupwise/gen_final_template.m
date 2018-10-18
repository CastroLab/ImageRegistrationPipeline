load('C:\Users\Administrator\ABA-Image-Registration\registration\groupwise\templates\groupwise_okno_25_sim_ms_subdiv_4_lambda_0.5_maxstep_50_gamma_7.5_maxcycle_50.mat')

filt = @(im) imgaussfilt(im, 1);
preprocess = @(im) standardizeImage(...
    filt(im));
% mean_image = @(im1, im2) mean(cat(3, im1, im2), 3);
compute_mean_image = @(im_stack) mean(im_stack, 3);

% Output image size
im_size = [480 480];
num_images = length(reg_images);
stack = zeros([im_size, num_images]);
rig_stack = stack;



for i = 1:num_images
    
    im = preprocess(imread(reg_images{i}));
    if size(im, 3) ~= 1
        im = rgb2gray(im);
    end
   
    stack(:, :, i) = im;
    rigid_aligned = imwarp(im, reg(i).tform);
    rig_stack(:, :, i) = imresize(rigid_aligned, im_size);

end

mirt_stack = mirt2Dgroup_transform(rig_im_stack, res);
template = mean(mirt_stack, 3);
imshow(template)
