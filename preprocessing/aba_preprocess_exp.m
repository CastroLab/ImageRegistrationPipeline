function [ image, test ] = aba_preprocess_exp( image,im_size )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

if ~exist('im_size','var')
     % second parameter does not exist, so default it to something
      im_size = 480;
end

exp_im = imread(image.exp_path);
cropped_exp = imcrop(exp_im, image.bb);
try
    cropped_exp = imresize(cropped_exp,[im_size,im_size]);
    test = 1;
catch 
    test = 0;
end

if test == 1
    image.cropped_exp = im2double(cropped_exp);
    image.masked_exp  = image.cropped_exp;
    image.masked_exp(image.mask) = 0;       
end

end

