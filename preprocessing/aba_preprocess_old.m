function [im] = aba_preprocess(im,im_size,filt_sigma)
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
dims = size(size(im));
if dims(2) == 3 
    im = rgb2gray(im);
end

im = imgaussfilt(im,filt_sigma); 
im = aba_imcrop_edge(im);
im = imresize(im,[im_size,im_size]);
%im = imadjust(im);
im = aba_mask_tissue(im);
im = im2double(im);
end

