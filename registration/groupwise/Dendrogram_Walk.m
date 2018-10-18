% Dendrogram Walk

%% Configuration
% -------------------------------------------------------------------------

% Data
conf.imDir ='/Users/dpaselti/ABA-Image-Registration/classification/aba_template_images/';
conf.dataDir = '/Users/dpaselti/ABA-Image-Registration/registration/groupwise/data/';
conf.numTrain = 15;
conf.numTest = 15;
conf.numClasses = 2;
conf.numWords = 1000;
conf.numSpatialX = [2 4];
conf.numSpatialY = [2 4];

% Vocab
conf.phowOpts = {'Step', 3};
conf.quantizer = 'kdtree';

% SVM
conf.svm.C = 10;
conf.svm.solver = 'sdca'; % Options = {'sdca', 'sgd', 'liblinear'}
conf.svm.biasMultiplier = 1;

% Misc.
conf.clobber = false;
conf.tinyProblem = false;
conf.randSeed = 1;

% Output
conf.prefix = 'template/1000_words';
split = strsplit(conf.prefix, '/');
dir_name = fullfile(conf.dataDir, split{1:end-1});
if ~exist(dir_name, 'dir')
    mkdir(dir_name);
end


if conf.tinyProblem
    conf.prefix = 'tiny';
    conf.numClasses = 5;
    conf.numSpatialX = 2;
    conf.numSpatialY = 2;
    conf.numWords = 300;
    conf.phowOpts = {'Verbose', 2, 'Sizes', 7, 'Step', 5};
end

conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']);
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']);
conf.distPath = fullfile(conf.dataDir, [conf.prefix '-dists.mat']);
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']);
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']);

randn('state',conf.randSeed);
rand('state',conf.randSeed);
vl_twister('state',conf.randSeed);

%%
load('1000_words-dists.mat');
load('/Users/dpaselti/ABA-Image-Registration/registration/groupwise/data/template/1000_words-model.mat')
images = model.images;
dendrogram_info = dists.Y;
Size = size(dendrogram_info);

% MIRT parameters
main.okno=30;            % mesh window size
main.similarity='ms';    % similarity measure 
main.subdivide = 5;      % number of hierarchical levels
main.lambda = 0.075;       % regularization weight, 0 for none
main.alpha=0.1;          % similarity measure parameter
main.single=1;           % don't show the mesh at each iteration

% MIRT Optimization parameters
optim.maxsteps = 40;    % maximum number of iterations (for a given frame being registered to the mean image)
optim.fundif = 1e-5;    % tolerance (for a given frame)
% optim.gamma = 10;      % initial optimization step size
optim.gamma = 1;      % initial optimization step size
optim.anneal=0.8;       % annealing rate
optim.imfundif=1e-6;    % Tolerance of the mean image (change of the mean image)
optim.maxcycle=30;      % maximum number of cycles (at each cycle all frames are registered to the mean image)

% a(a==0)=nan;             % set zeros to nans, the black color here corresponds to the border
                         % around the actual images. We exclude this border by
                         % setting it to NaNs.

mirt.main = main;
mirt.optim = optim;

% reg_im_path = '/Users/dpaselti/ABA-Image-Registration/classification/aba_template_images/';

for i = 1:Size(1)
    idx_1 = dendrogram_info(i,1);
    idx_2 = dendrogram_info(i,2);
    name_1 = fullfile(conf.imDir, images{idx_1}); 
    name_2 = fullfile(conf.imDir, images{idx_2});
    names = {name_1, name_2};
    %Find number of parent images
    m = aba_num_parent_images(name_1);
    n = aba_num_parent_images(name_2);
    num = m+n;
    weight1 = m/num;
    weight2 = n/num;
        
    %Call Groupwise function from Alex (name_1, name_2, mirt)
    reg_ims = aba_groupwise_reg(names, mirt);
    reg_ims(:,:,1) = reg_ims(:,:,1).*weight1;
    reg_ims(:,:,2) = reg_ims(:,:,2).*weight2;
    avg_im = reg_ims(:,:,1) + reg_ims(:,:,2);
    %avg_im = mean(reg_ims,3);
    
    % Save image
    base_name = fullfile('reg', strcat('mean_of_', int2str(idx_1), '_and_', ...
        int2str(idx_2),'_node_', int2str(Size(1)+i + 1), '_', num2str(num),'_','.jpg'));
    full_filename = fullfile(conf.imDir, base_name);
    images{end + 1} = base_name;
    imwrite(avg_im ,full_filename)
    
end