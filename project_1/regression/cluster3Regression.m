function [beta, Xtest, XTr] = cluster3Regression(yTr, XTr, Xte, clusters)

cluster = 3;
Xtrain = XTr(clusters(:,cluster), :);
Ytrain = yTr(clusters(:, cluster), :);


% Select relevant feature for the cluster 1. Determined with the correlation
% between the X belonging to this cluster and Y.
selected = [3, 4, 6, 21, 38, 47, 51, 52, 56, 67, 71];

% filter categorical and non-categorical variable
catVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];
A = (1:73);
Xall = A(~ismember(A, catVar));

% Normalize the non-categorical variables and dummy encode categorical
% variables

Xtrall = [normalize(Xtrain(:, Xall)), dummyEncode(Xtrain)];
extXTr = [normalize(XTr(:, Xall)), dummyEncode(XTr)];
%Model 3: RidgeRegression with all features and feature transformation and dummy encoded cat.
%variables
lambda = 5;
x56 = Xtrain(:, 56).^2;
x38 = Xtrain(:, 38).^2;
tXTr = [ones(length(Ytrain), 1) Xtrall normalize(x38) normalize(x56)];
beta = ridgeRegression(Ytrain, tXTr, lambda);

x56te = Xte(:, 56).^2;
x38te = Xte(:, 38).^2;
x56tr = XTr(:, 56).^2;
x38tr = XTr(:, 38).^2;
Xtest = [Xte normalize(x38te) normalize(x56te)];

XTr = [extXTr normalize(x38tr), normalize(x56tr)];
end