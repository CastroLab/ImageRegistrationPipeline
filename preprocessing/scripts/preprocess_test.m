%Preprocess Test

path = '/Volumes/etna/Scholarship/Jason Castro/Group/NSF Project/Images/image_grab_2016-09-16_11-53-41/';
load('/Volumes/etna/Scholarship/Jason Castro/Group/NSF Project/Images/image_name_list.mat');

%directory = dir(path);
%filenames = {directory.name};
%bad_names = filenames(cellfun('isempty', strfind(filenames, 'ish')));
%good_names = setdiff(filenames,bad_names);

num_sample = 100;
names = datasample(good_names,num_sample);

org_im_array = zeros(1000,1000,1,num_sample);
new_im_array = zeros(480,480,1,num_sample);

for i=1:num_sample
    org_image = imread(strcat(path,names{i}));
    image = rgb2gray(org_image);
    image = imresize(image,[1000,1000]);
    new_image = aba_preprocess(image,480,1);
    org_im_array(:,:,:,i) = im2double(image);
    new_im_array(:,:,:,i) = im2double(new_image);
end

figure, montage(org_im_array)
figure, montage(new_im_array)