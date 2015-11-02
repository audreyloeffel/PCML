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
K = 2;

% split data in K fold
setSeed(1);

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
    XTe = X(idxTe,:);
    yTr = y_train(idxTr);
    XTr = X(idxTr,:);    
    
    %global
    tXTr = [ones(length(yTr), 1) XTr];    
    tXTe = [ones(length(yTe), 1) XTe];    
    beta = ridgeRegression(yTr, tXTr, lambda);
    RMSETr(k) = sqrt(2*MSE(yTr, tXTr, beta));
    RMSETe(k) = sqrt(2*MSE(yTe, tXTe, beta));  
   
    
end

meanRMSETr = mean(RMSETr);
meanRMSETe = mean(RMSETe);

disp(meanRMSETr);
disp(meanRMSETe);

