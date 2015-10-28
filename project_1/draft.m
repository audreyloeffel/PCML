clear all;
load('Mumbai_regression.mat');


normX = normalize(X_train);
categoricalVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];


%% plot normalized features one by one
% figure;
% for i = 1:size(normX,2)
%    scatter(normX(:,i), y_train, '.');
%    pause; 
% end

%% plot normalized features
% figure;
% for i = 1:size(normX,2)
%    scatter(normX(:,i), y_train, '.');
%    hold on;
% end

%% Boxplot with normalized value X_train
figure;
boxplot(normX);

%% Boxplotwith unnormalized value X_train
figure;
boxplot(X_train);

% %% test with meanRegression and see the MSE -> crossValidation
% 
% K = 5; %nb de groupe
% N=size(y_train, 1); %nb de data
% npg = floor(N/5); %nb de data dans chaque groupe
% index = randperm(N);
% 
% for k = 1:K   
%     xTe=X_train(index(1+(k-1)*npg:k*npg), :);
%     yTe= y_train(index(1+(k-1)*npg:k*npg));
%     xTr=X_train(index([1:k-1*npg k*npg+1:end]), :);
%     yTr=y_train(index([1:k-1*npg k*npg+1:end]));
% end
% 
% % 4 groupes sont utilisé pour le training et le 5eme pour tester
% 

%%
% split data in K fold (we will only create indices)
degree = 1;
setSeed(1);
K = 4;
N = size(y_train,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% lambda values (INSERT CODE)
lambda = logspace(-2, 2, 100);

% K-fold cross validation
for i = 1:length(lambda)
	for k = 1:K
		% get k'th subgroup in test, others in train
		idxTe = idxCV(k,:);
		idxTr = idxCV([1:k-1 k+1:end],:);
		idxTr = idxTr(:);
		yTe = y_train(idxTe);
		XTe = X_train(idxTe,:);
		yTr = y_train(idxTr);
		XTr = X_train(idxTr,:);
        
        % beta + cost
       % tXTr = [ones(length(yTr),1) myPoly(XTr, degree)];
        tXt = [XTr ones(length(XTr),1)];
       % tXTe = [ones(length(XTe),1) myPoly(xTr, 1)];
        Xpoly = myPoly(XTr,degree);
        tX = [ones(length(y),1) Xpoly];
        
        
        beta = ridgeRegression(yTr, tXt, lambda(i));
        mse(i) = computeCost(yTr, tXt, beta);
        
        
    end
    
    mse2 = mean(mse);
    disp(mse2);
end

        
