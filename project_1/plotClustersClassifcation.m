load('Mumbai_regression.mat');
load('catClusters.mat');

figure;
X = normalize(X_train(clusters(:,1),:));
for i = 1:size(X,2)
   scatter(X(:,i), y_train(clusters(:,1)), '.', 'c');
   hold on;
end

X = normalize(X_train(clusters(:,2),:));
for i = 1:size(X,2)
   scatter(X(:,i), y_train(clusters(:,2)),'.', 'y');
   hold on;
end
X = normalize(X_train(clusters(:,3),:));
for i = 1:size(X,2)
   scatter(X(:,i), y_train(clusters(:,3)),'.', 'm');
   hold on;
end

title('Cluster Classification');
xlabel('X_train');
ylabel('y_train');