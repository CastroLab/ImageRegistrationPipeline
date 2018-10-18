%% Load Images from Query and preprocess 

clc, clear all

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

figure, montage(im_stack_cropped)
figure, montage(im_stack_masked)

%% Choose Images to eliminate due to preprocess failure

bad_idx = [4,9,25,31];
good_masked = im_stack_masked;
good_masked(:,:,:,bad_idx) = [];
figure, montage(good_masked)

mirt_images = squeeze(good_masked); %MIRT cannot handel the singleton dimension montage() requires. We remove it here.
%mirt_images = mirt_images(:,:,[1:5]);

%% Prepare MIRT

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
optim.fundif = 1e-5;    % tolerance (for a given frame)
% optim.gamma = 10;      % initial optimization step size
optim.gamma = 7.5;      % initial optimization step size
optim.anneal=0.8;       % annealing rate

optim.imfundif=1e-6;    % Tolerance of the mean image (change of the mean image)
optim.maxcycle=25;      % maximum number of cycles (at each cycle all frames are registered to the mean image)

%mirt_images(mirt_images==0)=nan;    % set zeros to nans, the black color here corresponds to the border
                % around the actual images. We exclude this border by
                % setting it to NaNs.

mirt.main = main;
mirt.optim = optim;

%% Run MIRT

res = mirt2Dgroup_sequence(mirt_images, main, optim);  % find the transformation (all to the mean)
b = mirt2Dgroup_transform(mirt_images, res);           % apply the transformation

%% Display Registration Results 
num_ims = size(mirt_images);
num_ims = num_ims(3);
reg_ims = zeros(im_size,im_size,1,num_ims);
reg_ims(:,:,1,:) = b;
figure, montage(reg_ims)

template = mean(b, 3);
figure, imshow(template)

%% Save Registration Results

%create path
dir_name = strcat('\query17_', date);
base_path = 'D:\METHODS_PAPER\registration\mirt\registration_results';
save_path = strcat(base_path, dir_name);

%make new directory
mkdir(save_path)

file_name = strcat(save_path,'\groupwise_reg_results.mat');
save(file_name,'mirt','im_stack_cropped','im_stack_masked','good_masked','reg_ims','template')

