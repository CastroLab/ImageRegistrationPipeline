function [I] = aba_mask_tissue(I)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Convert to grayscale
% I = im2double(I);
dims = size(size(I));
if dims(2) == 3
    I = rgb2gray(I);
end

% Subtract background and contrast enhance
background = imopen(I, strel('disk', 20));
background = ~background;
sub_image = I - im2uint8(background);
sub_image = imadjust(sub_image);

% 2-D adaptive noise-removal filtering
sub_image = wiener2(sub_image,[5 5]);

% Compute and smooth image histogram
pixelCounts = imhist(sub_image);
spc = smooth(pixelCounts);

% Find local minumums
DataInv = 1.01*max(spc) - spc;
[~, minIdx] = findpeaks(DataInv);

% Restrict local minimum domain
check_above = 210;
upper_limit = 248;
minIdx(minIdx < check_above) = [];
minIdx(minIdx > upper_limit) = [];

% Find lowest local min in domain of interest
threshold = min(minIdx);

% Set all pixels below threshold to black
if ~isempty(threshold)
    sub_image(sub_image > threshold-5) = 0;
end
% Compute and smooth final mask
bw = ~im2bw(sub_image, graythresh(sub_image));
bw = im2double(bw);
bw = imerode(bw,strel('disk',3));
bw = im2bw(imgaussfilt(bw, 30), graythresh(bw));
%bw = bwareaopen(bw, 60000);

I = imadjust(I);
I(bw) = 0;
end

