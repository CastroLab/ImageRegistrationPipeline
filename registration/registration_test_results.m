%% Load CNN Distances and Registration Check

load('D:\METHODS_PAPER\registration\mirt\subset_templates\test_templates\cnn_dists.mat')
load('D:\METHODS_PAPER\registration\mirt\subset_templates\test_templates\reg_check.mat')
load('D:\METHODS_PAPER\registration\mirt\subset_templates\test_templates\reg_ims_ish.mat')
load('D:\METHODS_PAPER\registration\mirt\subset_templates\test_templates\image_paths.mat')

%% Summary Statistics 
template_idx = [4,10,6,3,9,12,5,1,7,11,2,8];
template_success = mean(reg_check).*100;
figure, bar(template_success(template_idx))
title('Successful Registration Rate of Each Template')
xlabel('Template')
ylabel('% Success')

AvgNumGoodTemplateReg = mean(sum(reg_check));

image_success = mean(reg_check,2);
figure, bar(image_success.*100)
title('Successful Registration Rate of Each Image')
xlabel('Image')
ylabel('% Success')

[num_reg_per_image,edges] = histcounts(sum(reg_check,2),11);
figure, bar([0:1:10],num_reg_per_image.*0.5)
title('Number of Successful Registrations')
xlabel('Number of Good Registrations')
ylabel('% Of Total Images')

AvgNumGoodImReg = mean(sum(reg_check,2));

%% Look at Failed Images 

failed_idx = find(image_success == 0);
failed_paths = im_paths_nn(failed_idx);
failed_paths = strrep(failed_paths,'.mat','.jpg');
failed_paths = strrep(failed_paths,'structs_new','ish_color');

failed_stack = zeros(480,480,3,numel(failed_idx));
for i = 1:numel(failed_idx)
   
    failed_stack(:,:,:,i) = im2double(imread(failed_paths{i}));
end

figure, montage(failed_stack)

%% Check Number of Single Registrations Contributed By Each Template

num_solo_reg = zeros(1,12);

num_good_regs = numel(find(image_success ~= 0 ));

for i = 1:12
   ModRegCheck = reg_check;
   ModRegCheck(:,i) = 0;
   num_solo_reg(i) = num_good_regs - numel(find(mean(ModRegCheck,2) ~= 0));
end

figure, bar(num_solo_reg)
xlabel('Template')
ylabel('Number of Solo Registrations')
title('Solo Registrations of Each Template')
%% CNN Similarity Metric (Global)

GoodRegIdx = find(reg_check == 1);
BadRegIdx = find(reg_check == 0);

GoodDists = dists(GoodRegIdx);
BadDists = dists(BadRegIdx);

figure, GoodHist = histogram(GoodDists.^3);
title('Histogram of CNN Distance for All Successful Registrations')
xlabel('CNN Distance')
ylabel('Count')

figure, BadHist = histogram(BadDists.^3);
title('Histogram of CNN Distance for All Failed Registrations')
xlabel('CNN Distance')
ylabel('Count')

MeanGoodDists = mean(GoodDists);
MeanBadDists = mean(BadDists);

figure
plot(GoodHist.Values)
hold on
plot(BadHist.Values)
title('Successfull Versus Failed Registration CNN Distance Distribution')
xlabel('CNN Distance')
ylabel('Count')
legend('Successfull', 'Failed')

%% CNN Similarity Metric By Template

num_template = 12;
template_idx = [4,10,6,3,9,12,5,1,7,11,2,8];
figure
hold on
hists = {};
edges = {};
for i = 1:num_template
   
    GoodRegIdx = find(reg_check(:,template_idx(i)) == 1);
    BadRegIdx = find(reg_check(:,template_idx(i)) == 0);

    GoodDists = dists(GoodRegIdx);
    BadDists = dists(BadRegIdx);

    [GoodHist,GoodEdges] = histcounts(GoodDists);

    [BadHist,BadEdges] = histcounts(BadDists);

    
    subplot(3,4,i)
    plot(GoodHist)
    hold on
    plot(BadHist)
    title(strcat('Template #', int2str(i)))
    %xlabel('CNN Distance')
    %ylabel('Count')
    
    hists(i,1) = {GoodHist};
    hists(i,2) = {BadHist};
    
    
    edges(i,1) = {GoodEdges};
    edges(i,2) = {BadEdges};
   
end
legend('Successfull', 'Failed')

%% Load Templates 

template_base_path = 'D:\METHODS_PAPER\registration\mirt\subset_templates\';
template_names = dir(template_base_path);
template_names = {template_names(1:12).name};
template_full_path = strcat(template_base_path,template_names);

templates = zeros(480,480,1,12);

for i = 1:12
   load(template_full_path{i});
   templates(:,:,:,i) = template; 
end

figure, montage(templates)
templates = squeeze(templates);

%% Try MI

num_template = 12;
num_im = 200;

% Main parameters
    main.okno=16;           % mesh window size
    main.similarity='MI';   % similarity measure 
    main.subdivide = 3;     % number of hierarchical levels
    main.lambda = 0.1;     % regularization weight, 0 for none
   % main.lambda = 1;     % regularization weight, 0 for none
    main.alpha=0.1;        % similarity measure parameter
    main.single=1;          % don't show the mesh at each iteration

MIsimilarity = zeros(num_im,num_template);


for i = 1:num_template
    template = templates(:,:,i);
    
    for j = 1:num_im
        current_im = reg_ims_ish(:,:,:,j,i);
        MIsimilarity(j,i) = mirt_MI(template, current_im, 1000);
    
    end
end


%% MI Similarity Metric (Global)

GoodRegIdx = find(reg_check == 1);
BadRegIdx = find(reg_check == 0);

GoodDists = MIsimilarity(GoodRegIdx);
BadDists = MIsimilarity(BadRegIdx);

figure, GoodHist = histogram(GoodDists);
title('Histogram of MI for All Successful Registrations')
xlabel('MI')
ylabel('Count')

figure, BadHist = histogram(BadDists);
title('Histogram of MI for All Failed Registrations')
xlabel('MI')
ylabel('Count')

MeanGoodMI = mean(GoodDists);
MeanBadMi = mean(BadDists);

figure
plot(GoodHist.Values)
hold on
plot(BadHist.Values)
title('Successfull Versus Failed Registration MI Distribution')
xlabel('MI')
ylabel('Count')
legend('Successfull', 'Failed')

