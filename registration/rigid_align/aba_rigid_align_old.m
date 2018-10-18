function [fixed, moving, rigid_aligned, info ] = aba_rigid_align(im1, im2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

fixed = im2double(~im2bw(im1, graythresh(im1)));
moving = im2double(~im2bw(im2, graythresh(im2)));

[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumIterations = 20;
optimizer.MaximumStepLength = 0.05;
[rigid_aligned, info] = imregister(moving, fixed, 'affine', optimizer, metric);

end

