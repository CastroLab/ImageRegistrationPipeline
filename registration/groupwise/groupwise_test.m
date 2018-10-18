%% Template Design Test (following Ramsden)

%% 0. Setup

% image_dir = ['/Volumes/etna/Scholarship/Jason Castro/Group/NSF Project',...
%     '/Images/range1_sample_435-445_template_images'];

image_dir = './images/small_sample';

%% 1. Preprocess images

% (a) Bacground subtraction

% (b) Thresholding

% (c) Image smoothing (gaussian filter)

[images, imarray, BWarray] = aba_preprocess(image_dir);
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

%% 2. Calculate pairwise distances

BWim_rows = permute(im2uint8(BWarray), [3, 1, 2]);
BWim_rows = reshape(BWim_rows, num_images, []);
D = pdist(BWim_rows, @pdist_KL);
h_tree = linkage(D);
dendrogram(h_tree);
SD = squareform(D);
[Y,IX] = sort(SD);

% nn_to_i = IX(2, i);
nn_to_1 = IX(2, 1);
template = imarray(:, :, :, 1);
% to_align = imarray(:, :, :, nn_to_1);
to_align = imarray(:, :, :, 2);

% BW = squeeze(BWarray);
% BW = permute(BW, [3, 1, 2]);
% BWim_rows = reshape(BWarray, num_images, []);
% a = BWim_rows(1,:);
% imshow(reshape(a, 2200, 2200))
% SD(logical(eye(size(SD)))) = inf;
% [~, idx] = min(SD);

%% 3. Rigid alignment (Ramsden used ImageJ plugin 'Align Image by Line ROI')

% We will use MATLAB's 'imregister' instead, wrapped by aba_rigid_align().

i = 2;

rigid_aligned_array = zeros(m, n, 1, num_images);

[fixed, moving, rigid_aligned, tform] = aba_rigid_align_new(template, to_align);
rigid_aligned_array(:, :, :, 1) = fixed;
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


%% 4. Prepare for MIRT

% rs = @(n) imresize(rigid_aligned_array(:, :, :, n), 0.1);
% rs1 = @(im) imresize(im, 0.1);
% a = arrayfun(rs1, rigid_aligned_array);
% a = arrayfun(rs, 1:num_images, 'UniformOutput', 'False');
a = zeros(m/10, n/10, num_images);
for i = 1: num_images
    a(:, :, i) = imresize(squeeze(rigid_aligned_array(:, :, :, i)), 0.1);
end
a = im2double(a);
a = a(:, :, 1:2);
% a = im2double(squeeze(rigid_aligned_array));


%% 4. MIRT Groupwise Image Registration
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
optim.maxcycle=30;      % maximum number of cycles (at each cycle all frames are registered to the mean image)

a(a==0)=nan;    % set zeros to nans, the black color here corresponds to the border
                % around the actual images. We exclude this border by
                % setting it to NaNs.
                
res=mirt2Dgroup_sequence(a, main, optim);  % find the transformation (all to the mean)
b=mirt2Dgroup_transform(a, res);           % apply the transformation
% see the result
for i=1:size(b,3), imshow(b(:,:,i)); drawnow; pause(0.1); end;


%% Scale tranformation

scaled_res.X = imresize(20*res.X, 10);
scaled_res.okno = res.okno;


for i = 1: num_images
    a(:, :, i) = imresize(squeeze(rigid_aligned_array(:, :, :, i)), 0.1);
end
a = im2double(a);

template = im2double(template);
bw =im2double(BWarray);
bw = bw(:, :, 1:2);
scaled_b = mirt2Dgroup_transform(bw, scaled_res);
imshow(scaled_b(:, :, 2));
%% 
figure
subplot(2,3,1)
imshow(template)
title('im1')
subplot(2,3,2)
imshow(to_align)
title('im2')
subplot(2,3,3)
imshow(rigid_aligned)
title('im2 rigid aligned to im1')
subplot(2,3,4)
imshow(mean(b,3));
title('mean image')
subplot(2,3,5)
imshow(b(:, :, 1))
title('im1 post MIRT')
subplot(2,3,6)
imshow(b(:, :, 2))
title('im2 post MIRT')


% for i = 2:num_images
%     
%     % Select next image to be registered
%     im_rigid = rigid_aligned_array(:, :, :, i);
%     im_rigid = imresize(im_rigid, 0.1);
%     
%     % Apply MIRT
%     [res, newim]=mirt2D_register(refim, im_rigid, main, optim);
%     
%     % Apply affine transformation to origial ISH
%     im = imwarp(reg(i).ISH, reg(i).tform, 'OutputView',...
%         imref2d(size(reg(i).fixed)));
%     
%     % Apply MIRT transformation to each color channel
%     im = im2double(im);
%     ISH_reg = aba_mirt2D_transform_color(im, res);
%     post_sim = aba_similarity_metric(im2uint8(template), im2uint8(ISH_reg),...
%         'Shannon');
%     
%     % Store output
%     reg(i).res = res;
%     reg(i).mirt_reg = newim;
%     reg(i).ISH_reg = ISH_reg;
%     reg(i).post_sim = post_sim;
% end
% 
% save('test_output/reg.mat', 'reg', '-v7.3');

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





