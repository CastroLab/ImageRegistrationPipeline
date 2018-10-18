%% Experiment with match vl_phow

%% Load images
usr_home = getenv('HOME');
aba_base_dir = fullfile(usr_home, 'aba');
root_im_dir = 'classification/aba_images/range2_sample_440-460/';

im_list = {'cropped_2006_A930021G21Rik_431_2184_457ishfull.jpg', ...
    'cropped_2503_Psmd4_442_2328_447ishfull.jpg'};

ims = cell(length(im_list),1);
out = cell(length(im_list),1);
sel = cell(length(im_list),1);

for i = 1:length(ims)
    ims{i} = imread(fullfile(aba_base_dir, root_im_dir, im_list{i}));
    ims{i} = imgaussfilt(ims{i});
end

%% PHOW

for i = 1:length(ims)
    ims{i} = single(rgb2gray(ims{i}));
    
    figure(i)
%     im = ims{i};
%     msk = edge(im);
%     se = strel('disk', 10);
%     msk = imdilate(msk, se);
%     im(~msk) = 0;
%     ims{i} = im;
    imagesc(ims{i})
    colormap gray;
    axis equal;
    axis off;
    axis tight
    hold on;
    
    out{i} = aba_detect_points(ims{i}, 'PHOW');
    
    prcnt_thresh = 95;
    thresh = prctile(out{i}.f(3,:), prcnt_thresh);
    out{i}.f = out{i}.f(:, out{i}.f(3,:) > thresh );
    
    numpoints = 5000;
    f_length = size(out{i}.f, 2);
    perm = randperm(f_length);
    
    if numpoints > f_length
        sel{i} = perm(1:end);
    else
        sel{i} = perm(1:numpoints);
    end

    out{i}.new_f = out{i}.f(:, sel{i});
    scatter(out{i}.new_f(1, :), out{i}.new_f(2, :), 10)
    
end

figure
subplot(1, 2, 1)
scatter(out{1}.new_f(1, :), out{1}.new_f(2, :), 10)
subplot(1, 2, 2)
scatter(out{2}.new_f(1, :), out{2}.new_f(2, :), 10)

%% Find Matches

matches = vl_ubcmatch(out{1}.new_f, out{2}.new_f);

%% Plot Matches

f1 = out{1}.new_f;
f2 = out{2}.new_f;
mpoints1 = matches(1, :);
mpoints2 = matches(2, :);

X = cat(2, f1(1, mpoints1)', f1(2, mpoints1)');
Y = cat(2, f2(1, mpoints2)', f2(2, mpoints2)');

aba_match_plot(ims{1}, ims{2}, X , Y);
figure
subplot(1, 2, 1)
scatter(f1(1, mpoints1), f1(2, mpoints1));
subplot(1, 2, 2)
scatter(f2(1, mpoints2), f2(2, mpoints2));

%% Register

X = cat(2, f1(1, mpoints1)', f1(2, mpoints1)');
Y = cat(2, f2(1, mpoints2)', f2(2, mpoints2)');

perm = randperm(min(size(X,1), size(Y,1)));

if numpoints > length(perm)
    select = perm(1:end);
else
    select = perm(1:numpoints);
end

X = X(select, :);
Y = Y(select, :);
%%

% Init full set of options %%%%%%%%%%
opt.method='nonrigid'; % use nonrigid registration
opt.beta=2;            % the width of Gaussian kernel (smoothness)
opt.lambda=8;          % regularization weight

opt.viz=1;              % show every iteration
opt.outliers=0.7;       % use 0.7 noise weight
opt.fgt=0;              % do not use FGT (default)
opt.normalize=1;        % normalize to unit variance and zero mean before registering (default)
opt.corresp=1;          % compute correspondence vector at the end of registration (not being estimated by default)

opt.max_it=100;         % max number of iterations
opt.tol=1e-10;          % tolerance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Transform, C]=cpd_register(X,Y, opt);

figure,cpd_plot_iter(X, Y); title('Before');
figure,cpd_plot_iter(X, Transform.Y, C);  title('After registering Y to X. And Correspondences');
