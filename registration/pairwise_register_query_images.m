%% Load Images from Query and preprocess 

clc, clear all

%load data from previous registration
load('D:\METHODS_PAPER\registration\mirt\registration_results\cluster_7_of_7_FullSigma_UnSharedCovariance03-Oct-2017\groupwise_reg_results.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use commented code below is .mat files with the desired image do not
%already exist
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
q_num = 17;
query_path = strcat('D:\METHODS_PAPER\nearest_neighbors\data\CNN\query_' , num2str(q_num)) ;
cd(query_path)

filenames = dir('*.jpg');
nfiles = length(filenames);

%preprocess paramenters 
filt_sigma = 1;
im_size = 480;

%Initialze image array

im_stack_cropped = zeros(im_size, im_size, 1, nfiles);
im_stack_masked = zeros(im_size, im_size, 1, nfiles);

%Load image and preprocess
for i = 1:nfiles
   disp(i)
   currentfilename = filenames(i).name;
   currentimage = imread(currentfilename);
   currentimage = imgaussfilt(currentimage,filt_sigma); 
   [cropped_im,~,masked_im,~] = aba_mask_tissue(currentimage, im_size);
   im_stack_cropped(:,:,:,i) = cropped_im;
   im_stack_masked(:,:,:,i) = masked_im;
   
end
%}
figure, montage(im_stack_cropped)
figure, montage(im_stack_masked)

%% Choose Images to eliminate due to preprocess failure
if exist('good_masked') == 0 
    bad_idx = [35,56,75,95];
    good_masked = im_stack_masked;
    good_masked(:,:,:,bad_idx) = [];
    figure, montage(good_masked)
end
mirt_images = squeeze(good_masked); %MIRT cannot handel the singleton dimension montage() requires. We remove it here.
%mirt_images = mirt_images(:,:,[1:5]);

%% Prepare MIRT
test_mirt = exist('mirt');
if test_mirt ~= 0
    % Main parameters
    main.okno=20;           % mesh window size
    main.similarity='ms';   % similarity measure 
    main.subdivide = 3;     % number of hierarchical levels
    % main.lambda = 0.01;     % regularization weight, 0 for none
    main.lambda = 0.1;     % regularization weight, 0 for none
    main.alpha=0.1;         % similarity measure parameter
    main.single=1;          % don't show the mesh at each iteration


    % Optimization parameters
    optim.maxsteps = 40;    % maximum number of iterations (for a given frame being registered to the mean image)
    optim.fundif = 1e-6;    % tolerance (for a given frame)
     %optim.gamma = 10;      % initial optimization step size
    optim.gamma = 1;      % initial optimization step size
    optim.anneal=0.9;       % annealing rate

    optim.imfundif=1e-6;    % Tolerance of the mean image (change of the mean image)
    optim.maxcycle=25;      % maximum number of cycles (at each cycle all frames are registered to the mean image)



    mirt.main = main;
    mirt.optim = optim;
end


%% Run MIRT

dim = size(mirt_images);
num_im = dim(3);
im_size = dim(1);

reg_ims = zeros(im_size,im_size,1,num_im);

figure
for i = 1:num_im
    ish = mirt_images(:,:,i);
    [res, reg_ish]=mirt2D_register(template,ish, mirt.main, mirt.optim);
    %reg_exp=mirt2D_transform(exp, res);
    reg_ims(:,:,:,i) = reg_ish;
end

%% Display Registration Results 

figure, montage(reg_ims)



%% Save Registration Results

%create path
dir_name = strcat('\query17_', date);
base_path = 'D:\METHODS_PAPER\registration\mirt\registration_results';
save_path = strcat(base_path, dir_name);

%make new directory
mkdir(save_path)

file_name = strcat(save_path,'\pairwise_reg_ms2_results.mat');
save(file_name,'mirt','im_stack_cropped','im_stack_masked','good_masked','reg_ims','template')

