clear all;
load('Mumbai_regression.mat');
load('catClusters.mat');

catVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];
A = (1:73);
nonCatVar = A(~ismember(A,catVar));

X_nonCat = normalize(X_train(:, nonCatVar));

X = [X_nonCat, toBinary()];
tXTr = [ones(length(X), 1) X];

lambda = 0.5;
alpha = 0.001;



%% cluster 1

Y1 = clusters(:,1);

beta1 = logisticRegression(Y1, tXTr, alpha);
print('[Cluster 1] Beta computed');
RMSETr = sqrt(2*MSE(Y1, tXTr, beta1));

%p1 = sigmoid(tXTr * beta1);

%% cluster 2

Y2 = clusters(:,2);
beta2 = logisticRegression(Y2, tXTr, alpha);
print('[Cluster 2] Beta computed');

% [mRMSETr, mRMSETe] = cVRidgeRegression(X2, Y2, 0, 0.5, 0, 0);
% fprintf('[Cluster 2] Ridge Regression: Training %.4f Test %.4f \n', mRMSETr, mRMSETe)

%p2 = sigmoid(tXTr * beta2);

%% cluster 3

Y3 = clusters(:,3);
beta3 = logisticRegression(Y3, tXTr, alpha);
print('[Cluster 3] Beta computed');

% [mRMSETr, mRMSETe] = cVRidgeRegression(X3, Y3, 0, 0.5, 0, 0);
% fprintf('[Cluster 3] Ridge Regression: Training %.4f Test %.4f \n', mRMSETr, mRMSETe)
 %p3 = sigmoid(tXTr * beta3);

%% prediction







