% I1 = rgb2gray(imread(I1_file));
% I2 = rgb2gray(imread(I2_file));
%
% I1 = single(I1);
%
% % [f, d] = vl_phow(I1);
% out = aba_detect_points(I1, 'PHOW');
% f = out.f; d = out.d;
% f = f(:, f(3,:) > 15 );
%
% numpoints = 10000;
% f_length = size(f, 2);
% perm = randperm(f_length);
%
% if numpoints > f_length
%     sel = perm(1:end);
% else
%     sel = perm(1:numpoints);
% end
%
% figure
% imagesc(I1)
% colormap gray;
% axis equal;
% axis off;
% axis tight
%
% hold on;
% scatter(f(1, sel), f(2, sel), 10)
%
% %%
%
% I1 = uint8(I1);
% [r,f] = vl_mser(I1,'MinDiversity',0.7,...
%     'MaxVariation',0.2,...
%     'Delta',10) ;
%
%
% f = vl_ertr(f) ;
% vl_plotframe(f) ;
% %%
%
% M = zeros(size(I1)) ;
% for x=r'
%     s = vl_erfill(I1,x) ;
%     M(s) = M(s) + 1;
% end
%
% figure(2) ;
% clf ; imagesc(I1) ; hold on ; axis equal off; colormap gray ;
% [c,h]=contour(M,(0:max(M(:)))+.5) ;
% set(h,'color','y','linewidth',3) ;
%
% %%
%
% I1 = single(I1);
% figure
% imshow(I1)
% hold on ;
% frames = vl_covdet(I1, 'method', 'HarrisLaplace');
% f = frames(:, frames(3,:) > 3 );
% vl_plotframe(f) ;