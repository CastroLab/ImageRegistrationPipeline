load('D:\Preprocess\test_ppImageMaster.mat')
numImages = length(ppImageMaster);
struct_paths = ppImageMaster(:,2);
struct_names = ppImageMaster(:,1);
save_path = 'D:\Preprocess\images\ish_color\';
org_im_path = 'D:\ABA\images\master\ish\';

im_size = 480;

for i = 1:numImages
    
    % Load preprocessed struct
    load(struct_paths{i});
    bb = image.bb;
    mask = image.mask;
    
    %Get file names
    filename = strrep(struct_names{i},'.mat','.jpg');
    org_filename = filename(14:end);
    org_path = strcat(org_im_path, org_filename);
    full_path = fullfile(save_path,filename);
    
    %Crop and resize image
    org_im = imread(org_path);
    im = imcrop(org_im,bb);
    im = imresize(im,[im_size,im_size]);
    
    %For one color channel at a time mask and adjust contrast.
    for j = 1:3
        chan = im(:,:,j);
        chan = imadjust(chan);
        chan(mask) = 0;
        im(:,:,j) = chan;
    end
    
    imwrite(im, full_path)
    disp(i)
end