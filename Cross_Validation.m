clearvars
load('D:\EPFL\Fall 2015\Pattern Classification and Machine Learning CS-433\project_1\datas\Mumbai_regression.mat');

% form tX (INSERT CODE)
x = X_train;
meanX = mean2(x);
x = x - meanX;
stdX = std2(x);
x = x./stdX;

y = y_train;
N = length(y);

% split data in K fold (we will only create indices)
setSeed(1);
K = 4;
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% lambda values (INSERT CODE)
lambda = 1:50:10000;

% K-fold cross validation
for i = 1:length(lambda)
	for k = 1:K
		% get k'th subgroup in test, others in train
		idxTe = idxCV(k,:);
		idxTr = idxCV([1:k-1 k+1:end],:);
		idxTr = idxTr(:);
		yTe = y(idxTe);
		XTe = x(idxTe,:);
		yTr = y(idxTr);
		XTr = x(idxTr,:);
	    
        %Do Feature Transformation of Matrix tX
        XTrF = XTr.^(1/6);
        
        tX = [ones(length(yTr),1) XTrF];
        tXe = [ones(length(yTe),1) XTe];
		% least squares (INSERT CODE)
        %beta = leastSquares(yTr,tX);
        beta = ridgeRegression(yTr,tX,lambda(i));
		% training and test MSE(INSERT CODE)
		%mseTrSub(k) = 0; 
        rmseTrSub(k) = rmse(yTr,tX,beta);
		% testing MSE using least squares
		%mseTeSub(k) = 0; 
        rmseTeSub(k) = rmse(yTe,tXe,beta);

	end
	rmseTr(i) = mean(rmseTrSub);
	rmseTe(i) = mean(rmseTeSub);
end

% plot
figure;
plot(lambda, rmseTr,lambda,rmseTe);
legend('Train Error','Test Error');