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

% Classify into the clusters
for i = 1:length(Xte)
        [~, model(i,1)] = max(probabilities(i, :));
end
models = [model(:,1)==1, model(:,1)==2, model(:,1)==3];

% Beta's for regression in which clusters
betaC1 = cluster1Regression(y_train, X_train, clusters);
XTe1 = Xte;
[betaC2, XTe2, ~] = cluster2Regression(y_train, X_train, Xte, clusters);
[betaC3, XTe3, ~] = cluster3Regression(y_train, X_train, Xte, clusters);
fprintf('betaCi computed \n');



%% TODO : prediction of X_train -> compute the error

%[rmseTr, rmseTe] = cVModelRegression(X_train, y_train, clusters);
%fprintf('[Model Regression] Training %.4f Test %.4f \n', rmseTr, rmseTe);

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
csvwrite('predictions_regression.csv', yTe); 

%% plot
figure;
    X = normalize(Xte(model(:,1)==1,:));
    for i = 1:size(X,2)
        scatter(X(:,i), yTe(model(:,1)==1), '.', 'c');
        hold on;
    end
    
    X = normalize(Xte(model(:,1)==2,:));
    for i = 1:size(X,2)
        scatter(X(:,i), yTe(model(:,1)==2),'.', 'y');
        hold on;
    end
    X = normalize(Xte(model(:,1)==3,:));
    for i = 1:size(X,2)
        scatter(X(:,i), yTe(model(:,1)==3),'.', 'm');
        hold on;
    end
    
    title('Cluster Classification');
    xlabel('X_test');
    ylabel('prediction of Y');
    

