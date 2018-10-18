function [cropped_im, bb, masked_im, mask] = aba_mask_tissue(im, im_size, filt_sigma)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% Convert to grayscale
% I = im2double(I);
dims = size(size(im));
if dims(2) == 3
    im = rgb2gray(im);
end
%% Filter and Contrast Enhance
I = im;
I = imgaussfilt(I,filt_sigma); 

sub_image = imadjust(I);

% 2-D adaptive noise-removal filtering
sub_image = wiener2(sub_image,[5 5]);

%% Compute and smooth image histogram
pixelCounts = imhist(sub_image);
spc = smooth(pixelCounts);

%% Find Threshold 

%Fixed Range Method

% Find local minumums
DataInv = 1.01*max(spc) - spc;
[~, minIdx] = findpeaks(DataInv);

% Restrict local minimum domain
check_above = 210;
upper_limit = 255;
minIdx(minIdx < check_above) = [];
minIdx(minIdx > upper_limit) = [];

% Find lowest local min in domain of interest
FR_threshold = min(minIdx);


% Elbow Method
Max = max(spc);
Max_idx = find(spc == Max);
first_pt = [1, spc(1),0];
max_pt = [Max_idx, Max,0];

dists = zeros(1,Max_idx);

for i = 1:Max_idx
    pt = [i, spc(i),0];
    dists(i) = point_to_line(pt, first_pt, max_pt);
end

EM_threshold = find(dists == max(dists));

if FR_threshold < EM_threshold
    threshold = FR_threshold;
else
    threshold = EM_threshold;
end

%% Apply Threshold
% Set all pixels below threshold to black
if ~isempty(threshold)
    sub_image(sub_image > threshold) = 0;
end

%% Compute and smooth final mask
bw = ~im2bw(sub_image, graythresh(sub_image));
bw = im2double(bw);
bw = imerode(bw,strel('disk',3));
bw = im2bw(imgaussfilt(bw, 30), graythresh(bw));
%bw = bwareaopen(bw, 60000);

%% Find Bouding Box, Crop, and Mask 
bb = regionprops(~bw,'BoundingBox');
bb = bb.BoundingBox;
cropped_im = imcrop(im, bb);
cropped_bw = imcrop(bw, bb);

cropped_im = imresize(cropped_im,[im_size,im_size]);
cropped_im = im2double(cropped_im);
mask = imresize(cropped_bw,[im_size,im_size]);

masked_im = imadjust(cropped_im);
masked_im(mask) = 0;
end

