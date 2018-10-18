%Preprocess all images in one directory(org_path) and save preprocessed
%version in another directory (new_path).

org_path = '/Volumes/etna/Scholarship/Jason Castro/Group/NSF Project/Nissl_Images/Nissl_Images/';
new_path = '/Volumes/etna/Scholarship/Jason Castro/Group/NSF Project/Nissl_Images_Preprocessed/Nissl_Images_Preprocessed/';
good_image_flag = 'nissl';


directory = dir(org_path);
filenames = {directory.name};
bad_names = filenames(cellfun('isempty', strfind(filenames, good_image_flag)));
good_names = setdiff(filenames,bad_names);
num_im = numel(good_names);



for i=1:num_im
    
    org_name = good_names{i};
    org_image = imread(strcat(org_path,org_name));
    dims = size(size(org_image));
    
    if dims(2) == 3 
        image = rgb2gray(org_image);
    end
    
    new_image = aba_preprocess(image,480,1);
    new_name = strcat(new_path,'preprocessed_',org_name);
    imwrite(new_image, new_name)
    
    disp(strcat(int2str(i),' of  ', int2str(num_im)))
    
end
