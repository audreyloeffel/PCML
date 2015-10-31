clear all;
load('Mumbai_regression.mat');
load('catClusters.mat');

catVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];
A = (1:73);
nonCatVar = A(~ismember(A,catVar));

X_nonCat = normalize(X_train(:, nonCatVar));
% normalize only uncategorical variables or all ?
X = [X_nonCat, toBinary()];


lambda = 0.5;
alpha = 1;
%% cluster 1

X1 = X(clusters(:,1),:);
Y1 = y_train(clusters(:,1),:);

[mRMSETr, mRMSETe] = cVRidgeRegression(X1, Y1, 0, 0.5, 0, 0);
fprintf('[Cluster 1] Ridge Regression: Training %.4f Test %.4f \n', mRMSETr, mRMSETe)

tXTr = [ones(length(Y1), 1) X1];
beta1 = logisticRegression(Y1, tXTr, alpha);
RMSETr = sqrt(2*MSE(Y1, tXTr, beta1));



%% cluster 2

X2 = X(clusters(:,2),:);
Y2 = y_train(clusters(:,2),:);
tXTr = [ones(length(Y2), 1) X2];
beta2 = ridgeRegression(Y2, tXTr, lambda);

[mRMSETr, mRMSETe] = cVRidgeRegression(X2, Y2, 0, 0.5, 0, 0);
fprintf('[Cluster 2] Ridge Regression: Training %.4f Test %.4f \n', mRMSETr, mRMSETe)

%% cluster 3

X3 = X(clusters(:,3),:);
Y3 = y_train(clusters(:,3),:);
tXTr = [ones(length(Y3), 1) X3];
beta3 = ridgeRegression(Y3, tXTr, lambda);

[mRMSETr, mRMSETe] = cVRidgeRegression(X3, Y3, 0, 0.5, 0, 0);
fprintf('[Cluster 3] Ridge Regression: Training %.4f Test %.4f \n', mRMSETr, mRMSETe)

%% prediction

tXtr = [ones(length(y_train), 1) X];
prediction = tXtr * beta1;


%p1 = sigmoid(prediction);





