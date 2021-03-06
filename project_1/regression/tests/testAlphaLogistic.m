clear all;
load('Mumbai_regression.mat');
load('catClusters.mat');

X = X_train;
tXTr = [ones(length(X), 1) X];
K=5;
space = logspace(-4,2,30);
space = 0:0.001:0.01;

for i = 1:length(space)
    
    alpha = space(i);
    %% cluster 1
    fprintf('[Step %d]alpha = %.4f\n',i, alpha);
    Y1 = clusters(:,1);
    [errTr1(i,1), errTe1(i,1)] = crossValidation(X_train, Y1, K, alpha, 0, 'lr', 0);
    fprintf('[Cluster 1] Training %.4f Test %.4f \n', errTr1(i,1), errTe1(i,1));
    
%     % cluster 2
%     
%     Y2 = clusters(:,2);
%     [errTr2(i,1), errTe2(i,1)] = crossValidation(X_train, Y2, K, alpha, 0, 'lr', 0);
%     
%     fprintf('[Cluster 2] Training %.4f Test %.4f \n', errTr2(i,1), errTe2(i,1));
%     
%     % cluster 3
%     
%     Y3 = clusters(:,3);
%     [errTr3(i,1), errTe3(i,1)] = crossValidation(X_train, Y3, K, alpha, 0, 'lr', 0);
%     fprintf('[Cluster 3] Training %.4f Test %.4f \n', errTr3(i,1), errTe3(i,1));
end
figure;
plot(space, errTr1, space, errTe1);
xlabel('alpha');
ylabel('RMSE');
legend('Training error', 'Test error');

