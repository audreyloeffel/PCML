clear all;
load('Mumbai_classification.mat');

selected = [1:28];

%Select only some features
X = X_train(:, selected);

% Transorm y {-1, 1} to {0, 1}
Y = (y_train+1)./2;

% Transform the datas
proportion = 0.8;
[XTr, yTr, XTe, yTe] = split(Y, X, proportion);

nbTraining = size(XTr, 1);
nbTest = size(XTe, 1);
nbFeature = size(XTr, 2);

XTrnorm = normalize(XTr);
XTenorm = normalize(XTe);

%% Logistic Regression and Penalized Logistic Regression

tXtr = [ones(nbTraining, 1), XTrnorm];
tXte = [ones(nbTest, 1), XTenorm];

alpha = 0.01;
beta = logisticRegression(yTr, tXtr,alpha);
cost = computeCostLogistic(yTe, tXte, beta);
disp(cost);

lambda = 5;
betaPen = penLogisticRegression(yTr, tXtr, alpha, lambda);
costPen = computeCostLogistic(yTe, tXte, betaPen);
disp(costPen);

%% prediction

pred = tXtr*beta;
probaY1 = sigmoid(pred); % return the probability for the point



