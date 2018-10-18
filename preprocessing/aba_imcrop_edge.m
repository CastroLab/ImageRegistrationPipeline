function [cropped_im, msk, bb] = aba_imcrop_edge(im)
%aba_imcrop Crops an aba image
%   Accepts an uncropped aba image and returns a cropped image containing
%   only the tissue section

% figure 
% imshow(im)

% Threshold and clear border
% msk = ~im2bw(im, graythresh(im));
msk = imadjust(im);
msk = edge(edge(msk, 'canny'), 'canny');
msk = imclearborder(msk);

% figure 
% imshow(msk)

filt = fspecial('average', 25);
msk = imfilter(im2double(msk), filt, 'replicate');
msk = imadjust(msk);

% figure
% imshow(msk)

% Dilation
se = strel('disk', 10);
msk = imdilate(msk, se);

% figure 
% imshow(msk)

msk = im2bw(msk, graythresh(msk));
% msk(msk >0) = 1;
% msk(msk~=1) = 0;
% figure
% imshow(msk)
% 
% % Erosion
% se1 = strel('disk',20);
% msk = imerode(msk, se1);
% 
% % figure
% % imshow(msk)
% 
% % Large Dilation & Small Erosion
% se2 = strel('disk', 20);
% % se2 = strel('disk',300);
% msk = imdilate(msk, se2);
% 
% figure
% imshow(msk)
% 
% % se3 = strel('disk', 10);
% se3 = strel('disk', 20);
% msk = imerode(msk, se3);
% % 
% % figure
% % imshow(msk)
% 
% Find largest cc (i.e. tissue section)
cc = bwconncomp(msk);
numPixels = cellfun(@numel, cc.PixelIdxList);
[~, idx] = max(numPixels);

if isempty(idx)
    cropped_im = im;
    return;
end
% Mask tissue
msk(:) = 0;
msk(cc.PixelIdxList{idx}) = 1;

% im(~msk) = 255;

% Find bounding box and crop
bb = regionprops(msk,'BoundingBox');
bb = bb.BoundingBox;
cropped_im = imcrop(im, bb);

end

