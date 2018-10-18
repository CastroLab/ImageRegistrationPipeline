%% Template Design Test (following Ramsden)

%% 0. Setup

image_dir = ['/Volumes/etna/Scholarship/Jason Castro/Group/NSF Project',...
    '/Images/range1_sample_435-445_template_images'];

%% 1. Preprocess images

% (a) Bacground subtraction

% (b) Thresholding

% (c) Image smoothing (gaussian filter)

[images, imarray] = aba_preprocess(image_dir);
[m, n, ~, num_images] = size(imarray);

% Take a quick look
montage(imarray)

% Prepare registration array of structs
reg = repmat(...
    struct(...
    'ISH',           [],...
    'fixed',         [],...
    'moving',        [],...
    'rigid_aligned', [],...
    'tform',         [],...
    'mirt_reg',      [],...
    'res',           [],...
    'pre_sim',       [],...
    'mid_sim',       [],...
    'post_sim',      [],...
    'ISH_reg',       []),...
    num_images, 1 );

%% 2. Rigid alignment (Ramsden used ImageJ plugin 'Align Image by Line ROI')

% We will use MATLAB's 'imregister' instead, wrapped by aba_rigid_align().

rigid_aligned_array = zeros(m, n, 1, num_images);
template = imarray(:, :, :, 1);

% Affine Tranformation
for i = 1:num_images
    
    % Calculate affine transformation and pre/mid similarity measures
    to_align = imarray(:, :, :, i);
    [fixed, moving, rigid_aligned, tform] = aba_rigid_align_new(template, to_align);
    rigid_aligned_array(:, :, :, i) = rigid_aligned;
    pre_sim = aba_similarity_metric(im2uint8(fixed), im2uint8(moving),...
        'Shannon');
    mid_sim = aba_similarity_metric(im2uint8(fixed), im2uint8(rigid_aligned),...
        'Shannon');
    
    % Store intermediate values in reg cell.
    reg(i).ISH = to_align;
    reg(i).fixed = fixed;
    reg(i).moving = moving;
    reg(i).rigid_aligned = rigid_aligned;
    reg(i).tform = tform;
    reg(i).pre_sim = pre_sim;
    reg(i).mid_sim = mid_sim;
    %     rigid_tforms{i} = tform;
end

figure
montage(rigid_aligned_array)
saveas(gcf,'rigid_aligned_montage.png')

%% 3. MIRT Pairwise Image Registration

refim = rigid_aligned_array(:, :, :, 1);
refim = imresize(refim, 0.1);

% Main settings
main.similarity='ssd';  % similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI
main.subdivide=4;       % use 3 hierarchical levels
main.okno=5;            % mesh window size
main.lambda = 0.4;    % transformation regularization weight, 0 for none
main.single=1;          % show mesh transformation at every iteration

% Optimization settings
optim.maxsteps = 200;   % maximum number of iterations at each hierarchical level
optim.fundif = 1e-6;    % tolerance (stopping criterion)
optim.gamma = 1;        % initial optimization step size
optim.anneal=0.8;       % annealing rate on the optimization step

for i = 2:num_images
    
    % Select next image to be registered
    im_rigid = rigid_aligned_array(:, :, :, i);
    im_rigid = imresize(im_rigid, 0.1);

    % Apply MIRT
    [res, newim]=mirt2D_register(refim, im_rigid, main, optim);
    
    % Apply affine transformation to origial ISH
    im = imwarp(reg(i).ISH, reg(i).tform, 'OutputView',...
        imref2d(size(reg(i).fixed)));

    % Apply MIRT transformation to each color channel
    im = im2double(im);
    ISH_reg = aba_mirt2D_transform_color(im, res);
    post_sim = aba_similarity_metric(im2uint8(template), im2uint8(ISH_reg),...
        'Shannon');

    % Store output
    reg(i).res = res;
    reg(i).mirt_reg = newim;
    reg(i).ISH_reg = ISH_reg;
    reg(i).post_sim = post_sim;
end

save('test_output/reg.mat', 'reg', '-v7.3');

%% Generate Histogram

histogram([reg.pre_sim], 30);
hold on
histogram([reg.post_sim], 30);
legend('pre-reg', 'post-reg')



% res is a structure of resulting transformation parameters
% newim is a deformed image
%
% you can also apply the resulting transformation directly as
% newim=mirt2D_transform(im, res);
% 
% figure,imshow(refim); title('Reference (fixed) image');
% figure,imshow(im_rigid);    title('Source (float) image');
% figure,imshow(newim); title('Registered (deformed) image');

%% Register original ISH images
% 
% im1 = im2double(imarray(:, :, :, 1));
% im2 = im2double(imarray(:, :, :, 2));
% 
% % Apply affine transformation to origial ISH
% im_rigid = imwarp(im2,rigid_tforms{2}, 'OutputView', imref2d(size(fixed)));
% 
% % Apply MIRT transformation to ISH
% % im2_reg = mirt2D_transform(im2_rigid, res);
% 
% % Apply MIRT transformation to each color channel
% 
% ISH_reg = aba_mirt2D_transform_color(im_rigid, res);
% % im2_reg = zeros(size(im2_rigid));
% % for i = 1:3
% %     im2_reg(:,:,i) = mirt2D_transform(im2_rigid(:,:,i), res);
% % end
% figure
% subplot(2,2,1)
% imshow(im1)
% title('Reference Image')
% subplot(2,2,2)
% imshow(im2)
% title('Moving Image')
% subplot(2,2,3)
% imshow(im_rigid)
% title('Affine Registered')
% subplot(2,2,4)
% imshow(ISH_reg)
% title('MIRT Registered')
% subplot(2,2,4)
% 
% saveas(gcf,'Barchart.png')

%% Pre and Post Registration Similarity
% 
% JUNK
    % im2_reg = zeros(size(im2_rigid));
    % for i = 1:3
    %     im2_reg(:,:,i) = mirt2D_transform(im2_rigid(:,:,i), res);
    % end
        
    % Apply MIRT transformation to ISH
    % im2_reg = mirt2D_transform(im2_rigid, res);
    




