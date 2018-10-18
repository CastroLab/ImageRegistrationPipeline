function [fixed, moving, rigid_aligned, info ] = aba_rigid_align_grayscale(im1, im2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

fixed = rgb2gray(im1);
moving = rgb2gray(im2);

[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumIterations = 20;
optimizer.MaximumStepLength = 0.05;
[rigid_aligned, info] = imregister(moving, fixed, 'affine', optimizer, metric);

end
