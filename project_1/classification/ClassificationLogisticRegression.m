% Written by Audrey Loeffel and Meryem M'hamdi, EPFL for PCML Fall 2015
% all rights reserved

% This code applies LogisticRegression to our classification data,
% predicts test errors using cross validation method and predicts output 
% for test data using this model 

clear all;
load('D:\EPFL\Fall 2015\Pattern Classification and Machine Learning CS-433\project_1\datas\Mumbai_classification.mat');

fprintf('\n<<<<<<<Using Logistic Regression>>>>>');

selected = [1:28];

%Select only some features
X = X_train(:, selected);

% Transorm y {-1, 1} to {0, 1}
Y = (y_train+1)./2;

% split data in K fold 
setSeed(1);
K = 4; 
N = size(y_train,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% K-fold cross validation

for k = 1:K
    fprintf('\nK=%d',k);
    % get k'th subgroup in test, others in train
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = Y(idxTe);
    XTe = X(idxTe,:);
    yTr = Y(idxTr);
    XTr = X(idxTr,:);

    nbTraining = size(XTr, 1);
    nbTest = size(XTe, 1);
    nbFeature = size(XTr, 2);
    
    % Normalizing our training and testing data
    XTrnorm = normalize(XTr);
    XTenorm = normalize(XTe);
    
    %Do Feature Transformation of Matrix 
    XTr = XTr.^2;
    
    XTe = XTe.^2; 

    tXtr = [ones(nbTraining, 1), XTrnorm];
    tXte = [ones(nbTest, 1), XTenorm];
    
    %% Applying the method Logistic Regression 
    alpha = 0.00005;
    beta = logisticRegression(yTr, tXtr,alpha); 
    
    %%Prediction for train data
    pred = tXte*beta;
    probaY1 = sigmoid(pred); % return the probability for the point
    
    yClass = zeros(nbTest,1);
    %Classification for train data
    for i = 1:nbTest
        if probaY1(i) > 0.5
            yClass(i,1)= 1;
        else
            yClass(i,1) = -1;
        end
    end
    fprintf('\nCost using RMSE is %d',rmse(yTe,tXte,beta));
    fprintf('\nCost using 0-1 Loss is %d',zero_one_loss(yTe,yClass));
    fprintf('\nCost using Log Loss is %d',logLoss(yTe,probaY1));
end

%% Prediction for test data
xtest = normalize(X_test);
tXte = [ones(length(xtest), 1), xtest];
pred = tXte*beta; % Prediction using logistic regression model
probaY1 = sigmoid(pred); % return the probability for all points in test data

ytest = zeros(length(xtest),1);
%Classifying the data using classes 1 and -1
for i = 1:length(xtest)
    if probaY1(i) > 0.5
        ytest(i,1)= 1;
    else
        ytest(i,1) = -1;
    end
end

