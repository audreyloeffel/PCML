clear all;
load('Mumbai_regression.mat');
load('catClusters.mat');

cluster = 2;
Xtrain = X_train(clusters(:,cluster), :);
Ytrain = y_train(clusters(:, cluster), :);

% Select relevant feature for the cluster 1. Determined with the correlation
% between the X belonging to this cluster and Y.
selected = [3, 21, 37, 42, 46, 52, 68, 28, 31, 41];

% filter categorical and non-categorical variable
catVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];
A = (1:73);
B = A(ismember(A, selected));
nonCatVar = B(~ismember(B, catVar));
Xall = A(~ismember(A, catVar));





% Normalize the non-categorical variables and dummy encode categorical
% variables
Xtr_nonCat = normalize(Xtrain(:, nonCatVar));
Xtr = [Xtr_nonCat, dummyEncode(Xtrain)];
Xtrall = [normalize(Xtrain(:, Xall)), dummyEncode(Xtrain)];


% TODO : test different regressions, with or without categorical, with or whitout feature transformation, for each one find the best parameters
% (alpha or lambda), compute RMSE, choose the best

%LeastSquare

%Model 0: Only for comparisation, it's not relevant
lambda = 5;
[rmseTr, rmseTe] = crossValidation(X_train, y_train, 0, lambda, 'rr', 0);
fprintf('[Model 0] Training %.4f Test %.4f \n', rmseTr, rmseTe);

%Model 1: RidgeRegression with selected features and dummy encoded
%categorical variables
lambda = 5;
[rmseTr, rmseTe] = crossValidation(Xtr, Ytrain, 0, lambda, 'rr', 0);
fprintf('[Model 1] Training %.4f Test %.4f \n', rmseTr, rmseTe);

%Model 2: RidgeRegression with all features and dummy encoded cat.
%variables
lambda = 5;

[rmseTr, rmseTe] = crossValidation(Xtrall, Ytrain, 0, lambda, 'rr', 0);
fprintf('[Model 2] Training %.4f Test %.4f \n', rmseTr, rmseTe);

%Model 3: RidgeRegression with all features and feature transformation and dummy encoded cat.
%variables
lambda = 5;
x46 = Xtrain(:, 46).^2;
[rmseTr, rmseTe] = crossValidation([Xtrall normalize(x46)], Ytrain, 0, lambda, 'rr', 0);
fprintf('[Model 3] Training %.4f Test %.4f \n', rmseTr, rmseTe);


      


