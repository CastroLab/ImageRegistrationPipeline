function [reg_im] = aba_groupwise_reg_pair(I_name, J_name, mirt, weight1, weight2)
%aba_groupwise_reg_pair Returns the groupwise registered mean image

filt = @(im) imgaussfilt(im, 1);
preprocess = @(im) standardizeImage(...
                filt(...
                aba_mask_tissue(im)));
mean_image = @(im1, im2) mean(cat(3, im1, im2), 3);

% Output image size
im_size = [480 480];

% Load images and compute mean
I = preprocess(imread(I_name));
J = preprocess(imread(J_name));

if size(I, 3) ~= 1
    I = rgb2gray(I);
end

if size(J, 3) ~= 1
    J = rgb2gray(J);
end

IJ = mean_image(I, J);

% Rigid align
[I_IJ.im, I_IJ.tf] = aba_rigid_align_new(I, IJ);
[J_IJ.im, J_IJ.tf] = aba_rigid_align_new(J, IJ);
I = imwarp(I, I_IJ.tf);
J = imwarp(J, J_IJ.tf);

% Remove black edges in background
 I_IJ.im(I_IJ.im == 0) = 255;
 J_IJ.im(J_IJ.im == 0) = 255;

% Crop and resize
[I_IJ.im, I_IJ.msk, I_IJ.bb] = aba_imcrop_edge(I);
[J_IJ.im, J_IJ.msk, J_IJ.bb] = aba_imcrop_edge(J);

% 
% I(~I_IJ.msk) = 255;
% J(~J_IJ.msk) = 255;
I = imcrop(I, I_IJ.bb);
J = imcrop(J, J_IJ.bb);

I_IJ.im = imresize(I_IJ.im, im_size);
J_IJ.im = imresize(J_IJ.im, im_size);
I = imresize(I, im_size);
J = imresize(J, im_size);

imstack = im2double(cat(3, I_IJ.im, J_IJ.im));
imstack(imstack ==0) = nan;
finalstack = im2double(cat(3, I, J));

res = mirt2Dgroup_sequence(imstack, mirt.main, mirt.optim);
reg_stack = mirt2Dgroup_transform(finalstack, res);

reg_im = mean_image(reg_stack(:, :, 1).*weight1, reg_stack(:, :, 2).*weight2);
end

