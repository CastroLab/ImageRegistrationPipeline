%% Clear
clear all
clc

%% Get Template Paths 

template_base_path = 'D:\ClosedMCL_Registered_Images_12Templates\templates\';
template_names = dir(template_base_path);
template_names = {template_names(1:12).name};
template_full_path = strcat(template_base_path,template_names);

num_template = numel(template_names);

%{
templates = zeros(480,480,1,12);

for i = 1:12
   load(template_full_path{i});
   templates(:,:,:,i) = template; 
end

figure, montage(templates)
templates = squeeze(templates);

%}
%% Load Preprocessed ImageMaster

ImageMaster_path = 'D:\Preprocess\Image_Master_pp_color_closed_mcl_subset.mat';
load(ImageMaster_path);
ImageMaster = ImageMaster_pp_color_closed_mcl_subset;
im_paths = ImageMaster(:,2);

num_im = numel(im_paths);


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

BaseSavePath = 'D:\ClosedMCL_Registered_Images_12Templates\template';

for j = 1:num_template
    
    
    load(template_full_path{j});
    
    SavePath = strcat(BaseSavePath,int2str(j),'\');


    figure
    for i = 1:num_im
        disp(strcat('Template: ', int2str(j),' Image: ',int2str(i)))
        load(im_paths{i})
        
        ish = image.masked_ish;
        exp = image.masked_exp;
        BW = imbinarize(ish,'adaptive');

        [res, reg_bw]=mirt2D_register(template,BW, mirt.main, mirt.optim);
        reg_ish = mirt2D_transform(ish, res);
        reg_exp = mirt2D_transform(exp, res);

        ishName = strsplit(image.ish_path,'\');
        ishName = ishName(end);
        reg_ishName = strcat('reg_',ishName);
        RegIshPath = strcat(SavePath,reg_ishName);
        
        expName = strsplit(image.exp_path,'\');
        expName = expName(end);
        reg_expName = strcat('reg_',expName);
        RegExpPath = strcat(SavePath,reg_expName);

        if (~isempty(ish))
            imwrite(reg_ish,RegIshPath{1})
        end
        
        if (~isempty(exp))
            imwrite(reg_exp,RegExpPath{1})
        end
        
    end
    


end
