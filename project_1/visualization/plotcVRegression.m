clear all;
load('Mumbai_regression.mat');
load('catClusters.mat');

%% OUTPUT
% [Cluster 1] Training 12.6504 Test 12.6243
% [Cluster 2] Training 12.5649 Test 12.5666
% [Cluster 3] Training 10.6140 Test 10.5540
%%

X = X_train;
tXTr = [ones(length(X), 1) X];
K=5;
space = logspace(-4,2,30);
%space = 0:0.0001:2;
alpha  = 0.0001;
%cluster 1
Y1 = clusters(:,1);
[errTr1, errTe1] = crossValidation(X_train, Y1, K, alpha, 0, 'lr', 0);
fprintf('[Cluster 1] Training %.4f Test %.4f \n', errTr1, errTe1);

% cluster 2

Y2 = clusters(:,2);
[errTr2, errTe2] = crossValidation(X_train, Y2, K, alpha, 0, 'lr', 0);

fprintf('[Cluster 2] Training %.4f Test %.4f \n', errTr2, errTe2);

% cluster 3

Y3 = clusters(:,3);
[errTr3, errTe3] = crossValidation(X_train, Y3, K, alpha, 0, 'lr', 0);
fprintf('[Cluster 3] Training %.4f Test %.4f \n', errTr3, errTe3);
%
% figure;
% plot(space, errTr1, space, errTe2);
% xlabel('alpha');
% ylabel('RMSE');
% legend('Training error', 'Test error');

