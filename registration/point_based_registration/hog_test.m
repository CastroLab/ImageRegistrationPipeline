usr_home = getenv('HOME');
aba_base_dir = fullfile(usr_home, 'aba');
root_im_dir = 'classification/aba_images/range2_sample_440-460/';

I1_file = fullfile(aba_base_dir,root_im_dir, ...
    'cropped_2006_A930021G21Rik_431_2184_457ishfull.jpg');

%%

% SE = strel('disk', 5);
% msk = imdilate(edge(I1), SE);
% I1(~msk) = 0;
cellSize = 1;
hog = vl_hog(I1, cellSize, 'verbose', 'numOrientations', 1) ;
imhog = vl_hog('render', hog, 'verbose', 'numOrientations', 1) ;
clf ; imagesc(imhog) ; colormap gray ;

figure
s = size(hog,3);
n = ceil(sqrt(s));
for i = 1:size(hog,3)
   subplot(n,n, i)
   imagesc(hog(:, :, i))
end