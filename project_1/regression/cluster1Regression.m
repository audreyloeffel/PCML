function [beta] = cluster1Regression(yTr, XTr, clusters)

cluster = 1;
Xtrain = XTr(clusters(:,cluster), :);
Ytrain = yTr(clusters(:, cluster), :);

% Select relevant feature for the cluster 1. Determined with the correlation
% between the X belonging to this cluster and Y.
selected = [1, 3, 11, 7, 11, 13, 20, 21, 27, 33, 43, 55, 57, 58, 63, 64, 66, 73];

% filter categorical and non-categorical variable
catVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];
A = (1:73);
%B = A(ismember(A, selected));
%nonCatVar = B(~ismember(B, catVar));
Xall = A(~ismember(A, catVar));

% Normalize the non-categorical variables and dummy encode categorical
% variables
%Xtr_nonCat = normalize(Xtrain(:, nonCatVar));
%Xtr = [Xtr_nonCat, dummyEncode(Xtrain)];
Xtrall = [normalize(Xtrain(:, Xall)), dummyEncode(Xtrain)];

%Model 2: RidgeRegression with all features and dummy encoded cat.
%variables
lambda = 5;
tXTr = [ones(length(Ytrain), 1) Xtrall];
beta = ridgeRegression(Ytrain, tXTr, lambda);
%[rmseTr, rmseTe] = crossValidation(Xtrall, Ytrain, 0, lambda, 'rr', 0);
%fprintf('[Cluster 1 Model 2] Training %.4f Test %.4f \n', rmseTr, rmseTe);

end