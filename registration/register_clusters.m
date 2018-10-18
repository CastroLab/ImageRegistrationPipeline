%% Load Cluster Data, Image Master, and Embedding 

load('D:\METHODS_PAPER\clustering\data\CNNfeat_subset_GMM\4cluster_results.mat')
load('D:\Preprocess\Image_Master_pp_color_closed_mcl_subset.mat');
load('D:\METHODS_PAPER\embedding\data\CNNembedding_pp_color_closed_mcl_subset.mat');
load('D:\METHODS_PAPER\registration\mirt\registration_results\subset_template_cluster3\template1.mat')

ImageMaster = ImageMaster_pp_color_closed_mcl_subset;
   

%% Sort Images by Cluster

num_clusts = gmfit.NumComponents;

for i = 1:num_clusts

    clust_idx{i} = find(clusterX == i);
    
end

%% Find Closest n Images to Centroid
num_im = 100;

for i = 1:num_clusts
    clust_pts = embedding(clust_idx{i},:);
    centroid = gmfit.mu(i,:);
    pts = vertcat(centroid,clust_pts);
    dists = pdist(pts);
    SD = squareform(dists);
    centroid_dists = SD(:,1);
    centroid_dists(1) = [];
    [~,dist_idx] = sort(centroid_dists);
    nn_idx = dist_idx(1:num_im);
    far_idx = dist_idx(numel(dist_idx)-num_im:numel(dist_idx));
    im_master_idx_nn = clust_idx{i}(nn_idx);
    im_master_idx_far = clust_idx{i}(far_idx);
    im_paths_nn{i} = ImageMaster(im_master_idx_nn,2);
    im_paths_far{i} = ImageMaster(im_master_idx_far,2);
end
%% Choose Random Images 
 num_im = 100;
 
 for i = 1:num_clusts
    all_im_paths = ImageMaster(:,2);
    im_paths_nn{i} = randsample(all_im_paths,num_im);
    
end


%% Load Images and Montage



for i = 1:num_clusts
    im_stack = zeros(480,480,3,num_im);
    
    for j = 1:num_im
        path = im_paths_nn{i}(j);
        path = strrep(path,'.mat','.jpg');
        path = strrep(path,'structs_new','ish_color');
        im = imread(path{1});
        im = imresize(im,[480,480]);
        im_stack(:,:,:,j) = im2double(im);
    end
    figure, montage(im_stack)
end
%% Select Cluster

cluster_num = 3;

mirt_images = zeros(480,480,1,10);
%im_idx = [1,3,5,7,13,16,20,24,29,33,37,41,43,47,51,54,57,80,76,94];
%template_paths = im_paths_nn{cluster_num}(im_idx);
template_paths = im_paths_nn{cluster_num};

for j = 1:numel(template_paths)
    path = template_paths{j};
    path = strrep(path,'.mat','.jpg');
    path = strrep(path,'structs_new','ish_color');
    im = imread(path);
    im = imresize(im,[480,480]);
    im = rgb2gray(im);
    mirt_images(:,:,:,j) = im2double(im);
end

figure, montage(mirt_images)

mirt_images = squeeze(mirt_images); %MIRT cannot handel the singleton dimension montage() requires. We remove it here.


%% Prepare MIRT Groupwise

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
    optim.anneal=0.9;       % annealing rate

    optim.imfundif=1e-6;    % Tolerance of the mean image (change of the mean image)
    optim.maxcycle=25;      % maximum number of cycles (at each cycle all frames are registered to the mean image)



%mirt_images(mirt_images==0)=nan;    % set zeros to nans, the black color here corresponds to the border
                % around the actual images. We exclude this border by
                % setting it to NaNs.

mirt.main = main;
mirt.optim = optim;

%% Run MIRT Groupwise

res = mirt2Dgroup_sequence(mirt_images, main, optim);  % find the transformation (all to the mean)
b = mirt2Dgroup_transform(mirt_images, res);           % apply the transformation

%% Display Registration Results Groupwise
im_size = 480;
num_ims = size(mirt_images);
num_ims = num_ims(3);
reg_ims = zeros(im_size,im_size,1,num_ims);
reg_ims(:,:,1,:) = b;
figure, montage(reg_ims)

template = mean(b, 3);
figure, imshow(template)

%% Overlap Images and template Groupwise

im_pairs = zeros(im_size,im_size,3,num_ims);

for i = 1:num_ims
   pair = zeros(im_size,im_size,3);
   pair(:,:,1) = reg_ims(:,:,1,i);
   pair(:,:,2) = template;
   pair(:,:,3) = pair(:,:,1);
   im_pairs(:,:,:,i) = pair;
end

figure, montage(im_pairs)







%% Save Registration Results Groupwise
%{
%create path
dir_name = strcat('\cluster_7_of_7_FullSigma_UnSharedCovariance', date);
base_path = 'D:\METHODS_PAPER\registration\mirt\registration_results';
save_path = strcat(base_path, dir_name);

%make new directory
mkdir(save_path)

file_name = strcat(save_path,'\groupwise_reg_results.mat');
save(file_name,'mirt','im_stack_cropped','im_stack_masked','good_masked','reg_ims','template')
%}

%% Prepare MIRT Pairwise

    % Main parameters
    main.okno=16;           % mesh window size
    main.similarity='MS';   % similarity measure 
    main.subdivide = 3;     % number of hierarchical levels
    main.lambda = 0.1;     % regularization weight, 0 for none
    %main.lambda = 5;     % regularization weight, 0 for none
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



    mirt.main = main;
    mirt.optim = optim;



%% Run MIRT Pairwise

dim = size(mirt_images);
num_im = dim(3);
im_size = dim(1);

reg_ims_ish = zeros(im_size,im_size,1,num_im);
reg_ims_bw = zeros(im_size,im_size,1,num_im);

%filt = imgaussfilt(template,5);
%bw_template = imbinarize(filt,'adaptive');

bw_template = imbinarize(template,'adaptive');

figure
for i = 1:num_im
    ish = mirt_images(:,:,i);
    BW = ish;
    %BW = imgaussfilt(ish,8);
    BW = imbinarize(BW,'adaptive');
    
    [res, reg_bw]=mirt2D_register(bw_template,BW, mirt.main, mirt.optim);
    
    reg_ish = mirt2D_transform(ish, res);
    reg_ims_bw(:,:,:,i) = reg_bw;
    reg_ims_ish(:,:,:,i) = reg_ish;
end

%% Display Registration Results Pairwise

figure, montage(reg_ims)


%% Overlap Images and template Pairwise
%num_im = 9;
im_pairs = zeros(im_size,im_size,3,num_im);

for i = 1:num_im
   pair = zeros(im_size,im_size,3);
   pair(:,:,1) = template;
   pair(:,:,2) = reg_ims_ish(:,:,1,i);
   pair(:,:,3) = pair(:,:,1);
   im_pairs(:,:,:,i) = pair;
end

figure, montage(im_pairs)

