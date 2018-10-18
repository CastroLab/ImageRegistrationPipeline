%% Load Image Master

ImageMaster  = getImageMaster();

ish_names = ImageMaster(:,3);
exp_names = ImageMaster(:,4);


%% Preprocess Images

save_path = 'D:\Preprocess\images\structs_new';
im_size = 480;
filt_sigma = 1;

aba_preprocess_group( ish_names(17989:end), exp_names(17989:end), im_size, filt_sigma, save_path )


