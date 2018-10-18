usr_home = getenv('HOME');
aba_base_dir = fullfile(usr_home, 'aba');
root_im_dir = 'classification/aba_images/range2_sample_440-460/';

I1_file = fullfile(aba_base_dir,root_im_dir, ...
    'cropped_2006_A930021G21Rik_431_2184_457ishfull.jpg');
I2_file = fullfile(aba_base_dir,root_im_dir, ...
    'cropped_2503_Psmd4_442_2328_447ishfull.jpg');

I1 = rgb2gray(imread(I1_file));
I2 = rgb2gray(imread(I2_file));

% points1 = detectHarrisFeatures(I1);
% points2 = detectHarrisFeatures(I2);

points1 = detectSURFFeatures(I1, 'MetricThreshold', 200);
points2 = detectSURFFeatures(I2, 'MetricThreshold', 200);

[features1, valid_points1] = extractFeatures(I1, points1);
[features2, valid_points2] = extractFeatures(I2, points2);

indexPairs = matchFeatures(features1,features2);

matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);

figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);