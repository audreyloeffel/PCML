%load('train.mat')
%
% figure;
%  %scatter(train.X_hog(:, 1), train.y, '.');
%    
% for i = 1:50
%     
%     scatter(train.X_hog(:, i), train.y, '.');
%     xlabel('X\_hog');
%     ylabel('y');
%     title('Features');
%     pause
% end


img = imread( sprintf('train/imgs/train%05d.jpg', 1) );
subplot(121);
imshow(img); % image itself

subplot(122);
feature = hog( single(img)/255, 17, 8);
im( hogDraw(feature) ); colormap gray;
axis off; colorbar off;
print -dpdf hog_feature_ex.pdf