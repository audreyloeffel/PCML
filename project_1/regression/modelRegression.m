clear all;
load('Mumbai_regression.mat');
load('catClusters.mat');

catVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];
A = (1:73);
nonCatVar = A(~ismember(A,catVar));

% Normalize the non-categorical variables and dummy encode categorical
% variables
Xtr_nonCat = normalize(X_train(:, nonCatVar));
Xtr = [Xtr_nonCat, dummyEncode(X_train)];
Xte_nonCat = normalize(X_test(:, nonCatVar));
Xte = [Xte_nonCat, dummyEncode(X_test)];

% Compute betas in order to classify X_test into one of the three clusters
alpha = 0.001;
[beta1, beta2, beta3] = clustersRegression(Xtr, alpha, clusters);

% Probabilities for X_test's to belong to one of the three clusters
tXte = [ones(length(Xte), 1) Xte];
p1 = sigmoid(tXte * beta1);
p2 = sigmoid(tXte * beta2);
p3 = sigmoid(tXte * beta3);
probabilities = [p1 p2 p3];

% Beta's for regression in which clusters
betaC1 = cluster1Regression(y_train, X_train);
XTe1 = Xte;
[betaC2, XTe2] = cluster2Regression(y_train, X_train, Xte);
[betaC3, XTe3] = cluster3Regression(y_train, X_train, Xte);
fprintf('betaCi computed \n');



%% TODO : prediction of X_train -> compute the error

%% Prediction for X_test
for i = 1:length(Xte)
    [M, model(i,1)] = max(probabilities(i, :));
   
    switch model(i,1)
        case 1
            tXte =  [1 XTe1(i, :)];
            yTe(i,1) = tXte * betaC1;
        case 2
            tXte =  [1 XTe2(i, :)];
            yTe(i,1) = tXte * betaC2;
        case 3
            tXte =  [1 XTe3(i, :)];
            yTe(i,1) = tXte * betaC3;
    end
    
end

