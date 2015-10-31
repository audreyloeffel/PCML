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

K = 11;

