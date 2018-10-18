ppPath = 'D:\Preprocess\images\structs_new\';
contents = dir(ppPath);
isdir = [contents(1:numel(contents)).isdir];
names = {contents(1:numel(contents)).name};
names = names(isdir == 0);

paths  = fullfile(ppPath,names);

ppImageMaster = horzcat(names', paths');

save('D:\Preprocess\new_ppImageMaster.mat', 'ppImageMaster')
