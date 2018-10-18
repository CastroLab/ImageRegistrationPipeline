function [rigid_aligned, tform, fixed, moving] = aba_rigid_align_new(moving, fixed)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if size(moving, 3) ~= 1
    moving = im2double(~im2bw(moving, graythresh(moving)));
end

if size(fixed, 3) ~= 1
    fixed = im2double(~im2bw(fixed, graythresh(fixed)));
end

[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumIterations = 100;
optimizer.MaximumStepLength = 0.01;
tform = imregtform(moving, fixed, 'affine', optimizer, metric);
rigid_aligned = imwarp(moving, tform);

end

