%% Load model and move it to the GPU
disp('Loading model and moving it to GPU...')
net = load_model();
net = vl_simplenn_move(net, 'gpu');

for i = 1:1%numBatches

    % Load batch of images
    disp(['Loading batch ', num2str(i), ' of ', num2str(numBatches), '...'])
    names = imNames(batches(:, i));

    tic
    imStack = vl_imreadjpeg(names(1), ...
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
