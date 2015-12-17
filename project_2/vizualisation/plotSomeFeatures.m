load('train.mat')

figure;
 %scatter(train.X_hog(:, 1), train.y, '.');
 x = train.X_hog;
 y = train.y;
for i = 1:50
   scatter(x(:,i), y,'.');
   hold on;
end