clear all;
load('Mumbai_regression.mat');
load('classificationIntoThreeClusters.mat');
load('coordCluster1.mat');
load('coordCluster3.mat');

categoricalVar = [9, 10, 12, 15, 31, 35, 39, 48, 49, 59, 65, 68];

isole1 = 3; %interessant feature to classify the first cluster
isole3 = 21; % same for the third cluster
nbPoint = length(X_train);
scatter(X_train(:, isole1), y_train);
hold on;
scatter(X_train(:, isole3), y_train);
clusters = ones(length(X_train), 1) * 2; %in cluster 2 by default

%identify the points in the
X1 = X_train(:, isole1);

for i = 1:nbPoint
    for j = 1:length(coordC1)
       if X1(i) == coordC1(j)
            clusters(i) = 1;
        end
    end
end


X3 = X_train(:, isole3);

for i = 1:nbPoint
    for j = 1:length(coordC3)
        if X3(i) == coordC3(j)
            clusters(i) = 3;
        end
    end
end

%correlation
nbFeature = size(X_train, 2);
for i = 1:nbFeature
    corrGlobal(i) = corr(X_train(:,i), y_train);
    corrCluster1(i) = corr(X_train(clusters == 1 ,i), y_train(clusters == 1));
    corrCluster2(i) = corr(X_train(clusters == 2, i), y_train(clusters == 2));
    corrCluster3(i) = corr(X_train(clusters == 3, i), y_train(clusters == 3));
end
figure;

c = [corrGlobal' corrCluster1' corrCluster2' corrCluster3']; 
% we can identify which feature we should select for the regression for each clusters
% We should apply the regression on each cluster, applying the regression
% on the whole data isn't relevant
bar(c);
title('Correlation');
legend('global','cluster 1','cluster 2','cluster 3');
xlabel('Feature');
ylabel('Correlation (X, Y)');
print -dpdf corr.pdf;

binX = toBinary();
