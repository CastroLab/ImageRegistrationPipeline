function [im_reg] = aba_mirt2D_transform_color(im, res )
%aba_mirt2D_transform_color apply_mirt2D_transform for all color channels

dim = size(im);
im_reg = zeros(dim);
for i = 1:dim(end)
    im_reg(:,:,i) = mirt2D_transform(im(:,:,i), res);
end


end

