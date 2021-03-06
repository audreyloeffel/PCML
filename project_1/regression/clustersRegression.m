function [beta1, beta2, beta3] = clustersRegression(X_train, alpha, clusters )
%CLUSTERREGRESSION Return beta for each clusters


X = X_train;
tXTr = [ones(length(X), 1) X];

%% cluster 1

Y1 = clusters(:,1);

beta1 = logisticRegression(Y1, tXTr, alpha);
fprintf('[Cluster 1] Beta computed \n');
RMSETr = sqrt(2*MSE(Y1, tXTr, beta1));

%p1 = sigmoid(tXTr * beta1);

%% cluster 2

Y2 = clusters(:,2);
beta2 = logisticRegression(Y2, tXTr, alpha);
fprintf('[Cluster 2] Beta computed \n');

% [mRMSETr, mRMSETe] = cVRidgeRegression(X2, Y2, 0, 0.5, 0, 0);
% fprintf('[Cluster 2] Ridge Regression: Training %.4f Test %.4f \n', mRMSETr, mRMSETe)

%p2 = sigmoid(tXTr * beta2);

%% cluster 3

Y3 = clusters(:,3);
beta3 = logisticRegression(Y3, tXTr, alpha);
fprintf('[Cluster 3] Beta computed \n');

% [mRMSETr, mRMSETe] = cVRidgeRegression(X3, Y3, 0, 0.5, 0, 0);
% fprintf('[Cluster 3] Ridge Regression: Training %.4f Test %.4f \n', mRMSETr, mRMSETe)
%p3 = sigmoid(tXTr * beta3);
end
