load('D:\Preprocess\new_ppImageMaster.mat')
numImages = length(ppImageMaster);
struct_paths = ppImageMaster(:,2);
struct_names = ppImageMaster(:,1);
save_path = ('D:\Preprocess\images\ish_new\');


for i = 1:numImages
    
    load(struct_paths{i});
    im = image.masked_ish;
    filename = strrep(struct_names{i},'.mat','.jpg');
    full_path = fullfile(save_path,filename);
    imwrite(im, full_path)
    disp(i)
end

