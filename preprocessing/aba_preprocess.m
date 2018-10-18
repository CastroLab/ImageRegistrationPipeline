function [preprocessed] = aba_preprocess(path,im_size,filt_sigma)
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

im = imread(path);

dims = size(size(im));
if dims(2) == 3 
    im = rgb2gray(im);
end
im = imgaussfilt(im,filt_sigma); 
[cropped_im, bb, masked_im, mask] = aba_mask_tissue(im, im_size);
preprocessed = struct(...
                    'ish_path',          path,...
                    'cropped_im',    cropped_im,...
                    'bb',            bb,...
                    'masked_im',     masked_im,...
                    'mask',          mask);

end

