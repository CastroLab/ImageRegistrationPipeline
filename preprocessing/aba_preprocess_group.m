function [  ] = aba_preprocess_group( ish_names, exp_names, im_size, filt_sigma, save_path )
%Uwrtyhjwrtgbnteyuj7riyrkm
%   Detailed explanation goes here

num_im = numel(ish_names);

for i = 1:num_im
    image = struct(...
                    'ish_path',         ish_names(i),...
                    'exp_path',         exp_names(i),...
                    'bb',               [],...
                    'mask',             [],...
                    'cropped_ish',      [],...
                    'masked_ish',       [],...
                    'cropped_exp',      [],...
                    'masked_exp',       []);
    
    [image, test1] = aba_preprocess_ish(image, im_size, filt_sigma);
    [image, test2] = aba_preprocess_exp(image, im_size);
    
    if test1 == 1 || test2 == 1 
        path_parts = strsplit(image.ish_path,'\');
        name = path_parts{numel(path_parts)};
        name = strcat('\preprocessed_', name);
        name = strrep(name, '.jpg', '.mat');
        full_path = strcat(save_path, name);
        save(full_path, 'image')
    end
    
    
    disp(strcat(num2str(i),' of ', 32, num2str(num_im)))
end

