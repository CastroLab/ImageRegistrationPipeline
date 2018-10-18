% MIRT2D_EXAMPLE1: Non-rigid 2D registration example 1 with SSD similarity
% measure. 

clear all; clc; close all;

raw_refim_name = '/Users/dpaselti/Desktop/1758_446.jpg';
raw_im_name = '/Users/dpaselti/Desktop/1756_444.jpg';
movie_name = '/Users/dpaselti/Desktop/pairwise_registration_video1.mp4';

 % MIRT parameters
main.okno       = 20;   % mesh window size
main.similarity = 'ms'; % similarity measure 
main.subdivide  = 4;    % number of hierarchical levels
main.lambda     = 0.0001;  % regularization weight, 0 for none
main.alpha      = 0.093;  % similarity measure parameter
main.single     = 1;    % don't show the mesh at each iteration

% MIRT Optimization parameters
optim.maxsteps = 75;    % maximum number of iterations (for a given frame being registered to the mean image)
optim.fundif   = 1e-10;  % tolerance (for a given frame)
optim.gamma    = 4;   % initial optimization step size
optim.anneal   = 0.9;   % annealing rate
optim.imfundif = 1e-10;  % Tolerance of the mean image (change of the mean image)
optim.maxcycle = 50;    % maximum number of cycles (at each cycle all frames are registered to the mean image)

mirt.main = main;
mirt.optim = optim;

%raw_refim = imread('/Volumes/etna/Scholarship/Jason Castro/Group/NSF Project/Images/range1_sample_435-445/cropped_73615572_LOC434236_469_442ishfull.jpg');
%raw_im = imread('/Volumes/etna/Scholarship/Jason Castro/Group/NSF Project/Images/range1_sample_435-445/cropped_74800973_Ptgs1_470_444ishfull.jpg'); 

%load images
raw_refim = imread(raw_refim_name);
raw_im = imread(raw_im_name); 

%preprocess images
refim = im2double(aba_preprocess(raw_refim,480,2));
im = im2double(aba_preprocess(raw_im,480,2));

[res, newim, frames]=mirt2D_register_movie(refim,im, main, optim);

mov = struct('cdata',[],'colormap',[]);
for i = 1:numel(frames)
    frame_data = frames(i).cdata;
    
    if isempty(frame_data) == 0
        mov = horzcat(mov,frames(i));
    end
end
mov(1) = [];

v = VideoWriter(movie_name, 'MPEG-4');
v.FrameRate =  4;
v.Quality = 100;
open(v)
writeVideo(v,mov)
close(v)

figure, imshowpair(refim,im,'falsecolor')
figure, imshowpair(refim,newim,'falsecolor')
figure, imshowpair(refim,im,'blend')
figure, imshowpair(refim,newim,'blend')


% res is a structure of resulting transformation parameters
% newim is a deformed image 
%
% you can also apply the resulting transformation directly as
% newim=mirt2D_transform(im, res);

figure,imshow(refim); title('Reference (fixed) image');
figure,imshow(im);    title('Source (float) image');
figure,imshow(newim); title('Registered (deformed) image');


