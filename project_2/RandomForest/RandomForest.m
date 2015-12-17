clear all;
load train/train.mat;

%Extract original dataset
X = train.X_hog;
X = normalize(X);
Y = train.y;
N = length(Y);

cvpart = cvpartition(Y,'holdout',0.3);
Xtrain = X(training(cvpart),:);
Ytrain = Y(training(cvpart),:);
Xtest = X(test(cvpart),:);
Ytest = Y(test(cvpart),:);

bag = fitensemble(Xtrain,Ytrain,'Bag',400,'Tree',...
    'type','classification');

figure;
plot(loss(bag,Xtest,Ytest,'mode','cumulative'));
xlabel('Number of trees');
ylabel('Test classification error');

[predtest,scores] = bag.predict(Xtest);
%scores_norm = norm_score(scores);
scores_final = replaceonezero(scores(:,2));
logloss(Ytest,scores_final)