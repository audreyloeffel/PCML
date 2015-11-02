clear all;
load('Mumbai_regression.mat');
load('catClusters.mat');
fprintf('[Cross Validation] Start k-fold \n');

% split data in K fold
setSeed(1);
K = 5;
N = size(y_train,1);
idx = randperm(N);
Nk = floor(N/K);

for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for k = 1:K
    
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = y_train(idxTe);
    XTe = X_train(idxTe,:);
    yTr = y_train(idxTr);
    XTr = X_train(idxTr,:);
    clst = clusters(idxTr,:);
          
    catVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];
    A = (1:73);
    nonCatVar = A(~ismember(A,catVar));
    
    % Normalize the non-categorical variables and dummy encode categorical
    % variables
    Xtr_nonCat = normalize(XTr(:, nonCatVar));
    Xtr = [Xtr_nonCat, dummyEncode(XTr)];
    Xte_nonCat = normalize(XTe(:, nonCatVar));
    Xte = [Xte_nonCat, dummyEncode(XTe)];
    
    % Compute betas in order to classify X_test into one of the three clusters
    alpha = 0.001;
    [beta1, beta2, beta3] = clustersRegression(Xtr, alpha, clst);
    
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
    %models = [model(:,1)==1, model(:,1)==2, model(:,1)==3];
    
    % Beta's for regression in which clusters
    betaC1 = cluster1Regression(yTr, XTr, clst);
    XTe1 = Xte;
    
    [betaC2, XTe2, XTr2] = cluster2Regression(yTr, XTr, Xte, clst);
    [betaC3, XTe3, XTr3] = cluster3Regression(yTr, XTr, Xte, clst);
    fprintf('[Cross Validation] BetaCi computed \n');
    
    
    
    
    
    %% Prediction for X_test
    
    for i = 1:length(Xte)
        [~, model(i,1)] = max(probabilities(i, :));
        
        switch model(i,1)
            case 1
                tXte =  [1 XTe1(i, :)];
                predTe(i,1) = tXte * betaC1;
            case 2
                tXte =  [1 XTe2(i, :)];
                predTe(i,1) = tXte * betaC2;
                
            case 3
                tXte = [1 XTe3(i, :)];
                predTe(i,1) = tXte * betaC3;
                
        end    
    end
    
    % Probabilities for X_test's to belong to one of the three clusters
    tXTr = [ones(length(Xtr), 1) Xtr];
    p1 = sigmoid(tXTr * beta1);
    p2 = sigmoid(tXTr * beta2);
    p3 = sigmoid(tXTr * beta3);
    probabilities = [p1 p2 p3];
    
    % Classify into the clusters
    for i = 1:length(Xtr)
        [~, model(i,1)] = max(probabilities(i, :));
    end
    models = [model(:,1)==1, model(:,1)==2, model(:,1)==3];
    
    for i = 1:length(Xtr)
        [~, model(i,1)] = max(probabilities(i, :));
        
        switch model(i,1)
            case 1
                tXTr = [1 Xtr(i ,:)];
                predTr(i,1) = tXTr * betaC1;
            case 2
                tXTr = [1 XTr2(i,:)];            
                predTr(i,1) = tXTr * betaC2;                
            case 3
                tXTr = [1 XTr3(i,:)];
                predTr(i,1) = tXTr * betaC3;
                
        end    
    end
    RMSETr(k) = RMSE(yTr, predTr(:,1));
    RMSETe(k) = RMSE(yTe, predTe(:,1));
    end
    
    
 

mRMSETr = mean(RMSETr');
mRMSETe = mean(RMSETe');

