%% Load Cluster Data, Image Master, and Embedding 

load('D:\METHODS_PAPER\clustering\data\CNNfeat_subset_GMM\4cluster_results.mat')
load('D:\Preprocess\Image_Master_pp_color_closed_mcl_subset.mat');
load('D:\METHODS_PAPER\embedding\data\CNNembedding_pp_color_closed_mcl_subset.mat');


ImageMaster = ImageMaster_pp_color_closed_mcl_subset;


%% Template Paths 

TemplatePaths{1} = 'D:\METHODS_PAPER\registration\mirt\registration_results\subset_template_cluster3\cluster3_pt1_template_6,717,-80,49.mat';
TemplatePaths{2} = 'D:\METHODS_PAPER\registration\mirt\registration_results\subset_template_cluster3\cluster3_pt2_template_-13,9,-90.mat';
TemplatePaths{3} = 'D:\METHODS_PAPER\registration\mirt\registration_results\subset_template_cluster3\cluster3_pt3_template_-42,74,-70,8.mat';

%% Sort Images by Cluster

num_clusts = gmfit.NumComponents;

for i = 1:num_clusts

    clust_idx{i} = find(clusterX == i);
    
end

%% Choose Random Images 
 num_im = 100;
 
all_im_paths = ImageMaster(clust_idx{3},2);
im_paths = randsample(all_im_paths,num_im);

 %% Load Images and Montage
 
 im_stack = zeros(480,480,1,num_im);


    for j = 1:num_im
        path = im_paths(j);
        path = strrep(path,'.mat','.jpg');
        path = strrep(path,'structs_new','ish_color');
        im = imread(path{1});
        im = imresize(im,[480,480]);
        im = rgb2gray(im);
        im_stack(:,:,:,j) = im2double(im);
    end
    figure, montage(im_stack)

 mirt_images = squeeze(im_stack); %MIRT cannot handel the singleton dimension montage() requires. We remove it here.   
%% Prepare MIRT 

 % Main parameters
    main.okno=16;           % mesh window size
    main.similarity='SSD';   % similarity measure 
    main.subdivide = 3;     % number of hierarchical levels
    main.lambda = 0.1;     % regularization weight, 0 for none
   % main.lambda = 1;     % regularization weight, 0 for none
    main.alpha=0.1;        % similarity measure parameter
    main.single=1;          % don't show the mesh at each iteration


    % Optimization parameters
    optim.maxsteps = 40;    % maximum number of iterations (for a given frame being registered to the mean image)
    optim.fundif = 1e-6;    % tolerance (for a given frame)
     %optim.gamma = 10;      % initial optimization step size
    optim.gamma = 1;      % initial optimization step size
    optim.anneal=0.8;       % annealing rate

    optim.imfundif=1e-6;    % Tolerance of the mean image (change of the mean image)
    optim.maxcycle=25;      % maximum number of cycles (at each cycle all frames are registered to the mean image)



%mirt_images(mirt_images==0)=nan;    % set zeros to nans, the black color here corresponds to the border
                % around the actual images. We exclude this border by
                % setting it to NaNs.

mirt.main = main;
mirt.optim = optim;

%% Run MIRT Pairwise

for j = 1:3

load(TemplatePaths{j})
figure, imshow(template)    
    
dim = size(mirt_images);
im_size = dim(1);

reg_ims_ish = zeros(im_size,im_size,1,num_im);
reg_ims_bw = zeros(im_size,im_size,1,num_im);


figure
for i = 1:num_im
    ish = mirt_images(:,:,i);
    BW = ish;
    
    BW = imbinarize(BW,'adaptive');
    
    [res, reg_bw]=mirt2D_register(template,BW, mirt.main, mirt.optim);
    
    reg_ish = mirt2D_transform(ish, res);
    reg_ims_bw(:,:,:,i) = reg_bw;
    reg_ims_ish(:,:,:,i) = reg_ish;
end



% Overlap Images and template Pairwise

im_pairs = zeros(im_size,im_size,3,num_im);

for i = 1:num_im
   pair = zeros(im_size,im_size,3);
   pair(:,:,1) = template;
   pair(:,:,2) = reg_ims_ish(:,:,1,i);
   pair(:,:,3) = pair(:,:,1);
   im_pairs(:,:,:,i) = pair;
end
figure, montage(im_pairs)
results{j} = im_pairs;
end

