clear all;
load('Mumbai_regression.mat');
load('catClusters.mat');

X = X_train;
tXTr = [ones(length(X), 1) X];
K=2;
space = logspace(-4,2,30);
%space = 0:0.0001:2;

for i = 1:length(space)
    
    alpha = space(i);
    %% cluster 1
    fprintf('[Step %d]alpha = %.4f\n',i, alpha);
    Y1 = clusters(:,1);
    [errTr(i,1), errTe(i,1)] = crossValidation(X_train, Y1, K, alpha, 0, 'lr', 0);
    fprintf('[Cluster 1] Training %.4f Test %.4f \n', errTr(i,1), errTe(i,1));
    
%     %% cluster 2
%     
%     Y2 = clusters(:,2);
%     [errTr(i,1), errTe(i,1)] = crossValidation(X_train, Y2, K, alpha, 0, 'lr', 0);
%     
%     fprintf('[Cluster 2] Training %.4f Test %.4f \n', errTr(i,1), errTe(i,1));
%     
%     %% cluster 3
%     
%     Y3 = clusters(:,3);
%     [errTr(i,1), errTe(i,1)] = crossValidation(X_train, Y3, K, alpha, 0, 'lr', 0);
%     fprintf('[Cluster 3] Training %.4f Test %.4f \n', errTr(i,1), errTe(i,1));
end
figure;
plot(space, errTr, space, errTe);
xlabel('alpha');
ylabel('RMSE');
legend('Training error', 'Test error');

