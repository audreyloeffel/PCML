function [beta, Xtest, XTr] = cluster2Regression(yTr, XTr, Xte, clusters)

cluster = 2;
Xtrain = XTr(clusters(:,cluster), :);
Ytrain = yTr(clusters(:, cluster), :);

% Select relevant feature for the cluster 1. Determined with the correlation
% between the X belonging to this cluster and Y.
selected = [3, 21, 37, 42, 46, 52, 68, 28, 31, 41];

% filter categorical and non-categorical variable
catVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];
A = (1:73);
Xall = A(~ismember(A, catVar));
%disp(Xall);

% Normalize the non-categorical variables and dummy encode categorical
% variables
Xtrall = [normalize(Xtrain(:, Xall)), dummyEncode(Xtrain)];
extXTr = [normalize(XTr(:, Xall)), dummyEncode(XTr)];
%disp(size(Xtrall));
%Model 3: RidgeRegression with all features and feature transformation and dummy encoded cat.
%variables
lambda = 5;
x46 = Xtrain(:, 46).^2;
tXTr = [ones(length(Ytrain), 1) Xtrall normalize(x46)];
beta = ridgeRegression(Ytrain, tXTr, lambda);


x46te = Xte(:, 46).^2;
x46tr = XTr(:, 46).^2;
Xtest = [Xte normalize(x46te)];
XTr = [extXTr normalize(x46tr)];
end