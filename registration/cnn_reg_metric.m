%% Load Image Master, Embedding, and Templates 


load('D:\Preprocess\Image_Master_pp_color_closed_mcl_subset.mat');
load('D:\METHODS_PAPER\embedding\data\CNNembedding_pp_color_closed_mcl_subset.mat')

ImageMaster = ImageMaster_pp_color_closed_mcl_subset;

template_base_path = 'D:\METHODS_PAPER\registration\mirt\subset_templates\';
template_names = dir(template_base_path);
template_names = {template_names(1:12).name};
template_full_path = strcat(template_base_path,template_names);

templates = zeros(480,480,1,12);

for i = 1:12
   load(template_full_path{i});
   templates(:,:,:,i) = template; 
end

figure, montage(templates)
templates = squeeze(templates);

%% Choose Sample Images 

num_im = 200;

all_im_paths = ImageMaster(:,2);
im_paths_nn = randsample(all_im_paths,num_im);


mirt_images = zeros(480,480,1,num_im);


for j = 1:num_im
    path = im_paths_nn{j};
    path = strrep(path,'.mat','.jpg');
    path = strrep(path,'structs_new','ish_color');
    im = imread(path);
    im = imresize(im,[480,480]);
    im = rgb2gray(im);
    mirt_images(:,:,:,j) = im2double(im);
end

figure, montage(mirt_images)

mirt_images = squeeze(mirt_images); %MIRT cannot handel the singleton dimension montage() requires. We remove it here.

%% Setup MatConvNet
run(fullfile(vl_rootnn, 'matlab', 'vl_setupnn'))
disp('Loading model and moving it to GPU...')
net = load_model();
net = vl_simplenn_move(net, 'gpu');
%% Setup MIRT

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


%% Run MIRT

im_size = 480;

dists = zeros(num_im,12);
reg_check = zeros(num_im,12);

reg_ims_ish = zeros(im_size,im_size,1,num_im,12);
reg_ims_bw = zeros(im_size,im_size,1,num_im,12);
    
im_pairs = zeros(im_size,im_size,3,num_im,12);

for j = 1:12
    
    template = templates(:,:,j);
    template_path = 'D:\METHODS_PAPER\registration\mirt\template.jpg';
    imwrite(template,template_path)
    
    tic
        imStack = vl_imreadjpeg({template_path}, ...
            'SubtractAverage', single(net.meta.normalization.averageImage), ...
            'NumThreads', 6, ...
            'Resize', [227 227], ...
            'Pack', ...
            'GPU', ...
            'Verbose');
        imStack = imStack{:};
     toc

        % Run the CNN.
        disp(['Making the forward pass on batch: ', num2str(i)])
        im_res = vl_simplenn(net, imStack);

        % Save 7th fully connected layer.
        template_feature_vector = gather(squeeze(im_res(end-2).x));
    


    figure
    for i = 1:num_im
        im = mirt_images(:,:,i);
        BW = imbinarize(im,'adaptive');

        [res, reg_bw]=mirt2D_register(template,BW, mirt.main, mirt.optim);
        reg_ish = mirt2D_transform(im, res);

        reg_ims_bw(:,:,:,i,j) = reg_bw;
        reg_ims_ish(:,:,:,i,j) = reg_ish;

        reg_im_path = 'D:\METHODS_PAPER\registration\mirt\reg_im.jpg';

        reg_im = reg_ish;

        imwrite(reg_im,reg_im_path)

        
        tic
        imStack = vl_imreadjpeg({reg_im_path}, ...
            'SubtractAverage', single(net.meta.normalization.averageImage), ...
            'NumThreads', 6, ...
            'Resize', [227 227], ...
            'Pack', ...
            'GPU', ...
            'Verbose');
        imStack = imStack{:};
        toc

        % Run the CNN.
        disp(['Making the forward pass on batch: ', num2str(i)])
        im_res = vl_simplenn(net, imStack);

        % Save 7th fully connected layer.
        reg_im_feature_vector = gather(squeeze(im_res(end-2).x));

        dists(i,j) = pdist([template_feature_vector, reg_im_feature_vector]');
        
        pair = zeros(im_size,im_size,3);
        pair(:,:,1) = template;
        pair(:,:,2) = reg_ims_ish(:,:,1,i);
        pair(:,:,3) = pair(:,:,1);
        im_pairs(:,:,:,i,j) = pair;
        
    end
    


end

%% Marke Pair Montages 
for j = 1:12
    for i = 1:num_im
        pair = zeros(im_size,im_size,3);
        pair(:,:,1) = templates(:,:,j);
        pair(:,:,2) = reg_ims_ish(:,:,:,i,j);
        pair(:,:,3) = pair(:,:,1);
        im_pairs(:,:,:,i,j) = pair;
    end
end

%% Reg Check

    figure
    hold on 
    for m = 5:12
        for k = 1:num_im
            
            imshow(im_pairs(:,:,:,k,m))
            check = input('Was the registration successfull? Respond 1 for yes and 0 for no.');
            reg_check(k,m) = check;
        end
    end