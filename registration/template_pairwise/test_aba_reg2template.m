%% Load images

usr_home = getenv('HOME');
aba_base_dir = fullfile(usr_home, 'aba');
root_im_dir = 'classification/aba_images/range2_sample_440-460/';

template = fullfile(aba_base_dir,...
    'registration/groupwise/templates/groupwise_15_images.png');
im_list = {'cropped_2006_A930021G21Rik_431_2184_457ishfull.jpg', ...
           'cropped_2503_Psmd4_442_2328_447ishfull.jpg'};

im_names = cell(length(im_list), 1);
for i = 1:length(im_names)
   im_names{i} = fullfile(aba_base_dir, root_im_dir, im_list{i}); 
end

mirt = aba_load_mirt_default();
[res, reg, reg_im] = aba_im2temp_reg(im_names{1}, template, mirt);