function [image, test] = aba_preprocess_ish(image,im_size,filt_sigma)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if ~exist('im_size','var')
     % second parameter does not exist, so default it to something
      im_size = 480;
end

if ~exist('filt_sigma','var')
     % third parameter does not exist, so default it to something
      filt_sigma = 1;
end

im = imread(image.ish_path);

dims = size(size(im));
if dims(2) == 3 
    im = rgb2gray(im);
end
%im = imgaussfilt(im,filt_sigma); 
try
    [image.cropped_ish, image.bb, image.masked_ish, image.mask] = aba_mask_tissue(im, im_size, filt_sigma);
    test = 1;
catch
    test = 0;
end

end

