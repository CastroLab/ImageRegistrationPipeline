% Load test data
conf.imDir ='/Users/dpaselti/ABA-Image-Registration/classification/aba_template_images/';
load('/Users/dpaselti/ABA-Image-Registration/registration/groupwise/data/template/1000_words-model.mat')
I = fullfile(conf.imDir, model.images{13});
J = fullfile(conf.imDir, model.images{16});


% MIRT parameters
main.okno=20;            % mesh window size
main.similarity='ms';    % similarity measure 
main.subdivide = 3;      % number of hierarchical levels
main.lambda = 0.075;       % regularization weight, 0 for none
main.alpha=0.1;          % similarity measure parameter
main.single=1;           % don't show the mesh at each iteration

% MIRT Optimization parameters
optim.maxsteps = 40;    % maximum number of iterations (for a given frame being registered to the mean image)
optim.fundif = 1e-5;    % tolerance (for a given frame)
% optim.gamma = 10;      % initial optimization step size
optim.gamma = 7.5;      % initial optimization step size
optim.anneal=0.8;       % annealing rate
optim.imfundif=1e-6;    % Tolerance of the mean image (change of the mean image)
optim.maxcycle=30;      % maximum number of cycles (at each cycle all frames are registered to the mean image)

% a(a==0)=nan;             % set zeros to nans, the black color here corresponds to the border
                         % around the actual images. We exclude this border by
                         % setting it to NaNs.

mirt.main = main;
mirt.optim = optim;


%% Register

regim_im = aba_groupwise_reg_pair(I, J, mirt);

%% 
figure
subplot(1, 3, 1)
imshow(I)
subplot(1, 3, 2)
imshow(J)
subplot(1, 3, 3)
imshow(reg_im)
