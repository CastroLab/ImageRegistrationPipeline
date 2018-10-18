%% Load TSNE Embedding and Plot

if ~exist('DATADIR')
    DATADIR = fullfile('D:\', 'METHODS_PAPER', 'embedding', 'data');
    FILENAME = fullfile('CNNembedding_pp_color.mat');
    load(fullfile(DATADIR, FILENAME))

    figure, scatter(embedding(:,1), embedding(:,2), '.')
end

X = embedding;

%% Load Image Master
load('D:\Preprocess\new_ppImageMaster.mat')

%% Grid Embedded Space

x_max = 100;
x_min = -100;
y_max = 100;
y_min = -100;
increment = 20;

X = [x_min:increment:x_max];
Y = [y_min:increment:y_max];

for i = 1:(numel(X)-1)
   for j = 1:(numel(Y)-1) 
       %Define Box
       box_coords = zeros(4,2);
       
       %bottom left
       box_coords(1,1) = X(i);
       box_coords(1,2) = Y(j);
       
       %bottom right
       box_coords(2,1) = X(i) + increment;
       box_coords(2,2) = Y(j);
       
       %top right
       box_coords(3,1) = X(i) + increment;
       box_coords(3,2) = Y(j) + increment;
       
       %top left
       box_coords(4,1) = X(i);
       box_coords(4,2) = Y(j) + increment;
      
       %Find all points in box
       box_pts = inpolygon(embedding(:,1), embedding(:,2), box_coords(:,1), box_coords(:,2));
       idx = find(box_pts == 1);
       
       
       %Select points to grab images of
       num_im = 100;
       if numel(idx) > num_im
           idx = randsample(idx,num_im);
       else
           num_im = numel(idx);
       end
       
       %Load Images
       im_size = 100;
       im_stack = zeros(im_size,im_size,3,num_im);
       
       im_paths = ppImageMaster(idx,2);
       im_paths = strrep(im_paths,'.mat','.jpg');
       im_paths = strrep(im_paths,'structs_new','ish_color');
       
       if num_im > 0
           for k = 1:num_im
               im = imread(im_paths{k});
               im = imresize(im,[im_size, im_size]);
               im_stack(:,:,:,k) = im2double(im);
           end
       end
       
       %Make Montage and Save
       figure, montage(im_stack)
       name = strcat(num2str(X(i)+0.5*increment), ', ', num2str(Y(j)+0.5*increment));
       title(name)
       save_path = 'D:\METHODS_PAPER\embedding\data\all_pp_color_montage\';
       name = strcat(name,'.jpg');
       full_path = strcat(save_path,name);
       saveas(gcf,full_path)
   end   
end