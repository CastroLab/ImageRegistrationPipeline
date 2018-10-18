im_size = [220 220];
im1 = zeros(im_size);
im2 = zeros(im_size);

center1 = [120, 80];
center2 = [120, 110];

im1(center1(1), center1(2)) = 255;
im2(center2(1), center2(2)) = 255;
se = strel('disk', 50);
im1 = im2uint8(imdilate(im1, se));
im2 = im2uint8(imdilate(im2, se));
figure
subplot(1, 2, 1)
imshow(im1)
subplot(1, 2, 2)
imshow(im2)

[res, reg, reg_im] = aba_pairwise_reg(im1, im2, aba_load_mirt_default);