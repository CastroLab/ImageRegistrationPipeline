%% Compute fc7 features using our train model output of ABA images

%% Setup MatConvNet
run(fullfile(vl_rootnn, 'matlab', 'vl_setupnn'))

%% Load images
imageMaster = getImageMaster();
numImages = length(imageMaster);
imNames = imageMaster(:, 3);

%% Prep batches
NUMCLASSES = 6;
BATCHSIZE = 512;

numBatches = floor(numImages / BATCHSIZE);
batches = reshape(1:(numImages - mod(numImages, BATCHSIZE)), BATCHSIZE, numBatches);

% Initialize cell array to store outputs
fullyConnectedLayer = cell(numBatches + 1, 1);  % +1 for final batch (which is less than 512)

%% Load model and move it to the GPU
disp('Loading model and moving it to GPU...')
net = load_model();
net = vl_simplenn_move(net, 'gpu');

for i = 1:numBatches

    % Load batch of images
    disp(['Loading batch ', num2str(i), ' of ', num2str(numBatches), '...'])
    names = imNames(batches(:, i));

    tic
    imStack = vl_imreadjpeg(names, ...
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
    fullyConnectedLayer{i} = gather(squeeze(im_res(end-2).x));
end

%% Get features for 
finalNames = imNames((62 * 512) + 1: end);
imStack = vl_imreadjpeg(finalNames, ...
    'SubtractAverage', single(net.meta.normalization.averageImage), ...
    'NumThreads', 6, ...
    'Resize', [227 227], ...
    'Pack', ...
    'GPU', ...
    'Verbose');
imStack = imStack{:};

% Run the CNN.
disp(['Making the forward pass on final batch'])
im_res = vl_simplenn(net, imStack);
fullyConnectedLayer{end} = gather(squeeze(im_res(end-2).x));
%% Concatenate 'fc7' features across batches
fc7Features = cat(2, fullyConnectedLayer{:})';

%% Save result
save(fullfile(get_root(), 'feature_extraction', 'data', 'featCNN1.mat'), 'fc7Features');

