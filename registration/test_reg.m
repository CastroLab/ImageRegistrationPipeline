figure, imshow(template)

filt = imgaussfilt(template,5);
figure, imshow(filt)

bw_template = imbinarize(filt,'adaptive');
figure, imshow(bw_template)