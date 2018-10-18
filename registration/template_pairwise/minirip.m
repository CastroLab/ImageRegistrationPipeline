%% Minirip.m
%  A small test rip on 1000 images.

% PATHS
usr_home = getenv('HOME');
aba_base_dir = fullfile(usr_home, 'aba');
out_dir = fullfile(aba_base_dir, 'registration/template_pairwise/minirip/');
overwrite = false;

template = fullfile(aba_base_dir, ...
    'registration/groupwise/templates/groupwise_15_images.png');
root_im_dir = fullfile('/Volumes/etna/Scholarship', ...
    'Jason Castro/Group/NSF Project/Images/937_images/937_images');

d = dir(fullfile(root_im_dir, '*.jpg'));
im_names = {d.name};
im_paths = cellfun(@(im) fullfile(root_im_dir, im), ...
    im_names, 'UniformOutput',false);

mirt = aba_load_mirt_default();

for i = 1:length(im_paths)
    
    out_file = fullfile(out_dir, strcat('reg_', im_names{i}, '.jpg'));
    if ~exist(out_file, 'file') || overwrite
        [~, ~, reg_im] = aba_im2temp_reg(im_paths{i}, template, mirt);
        imwrite(reg_im, out_file)
    end
end
